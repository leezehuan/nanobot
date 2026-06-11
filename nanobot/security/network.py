"""网络安全辅助函数：防 SSRF 与内网地址识别。

【中文名称】网络目标安全校验

当模型准备访问 URL、调用抓取工具或执行带 URL 的命令时，
我们需要避免它误访问：

- 本机回环地址
- 私有网段
- 云厂商元数据地址
- 其它内部网络资源

否则就可能形成 SSRF（服务端请求伪造）风险。
"""

from __future__ import annotations

import ipaddress
import re
import socket
from contextlib import suppress
from urllib.parse import urlparse

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),   # 运营商级 NAT 网段
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # 链路本地地址 / 云元数据常用网段
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),          # IPv6 本地唯一地址
    ipaddress.ip_network("fe80::/10"),         # IPv6 链路本地地址
]

_URL_RE = re.compile(r"https?://[^\s\"'`;|<>]+", re.IGNORECASE)

_allowed_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []


def configure_ssrf_whitelist(cidrs: list[str]) -> None:
    """配置 SSRF 白名单网段。

    少数部署场景下，某些“看起来像内网”的网段其实是用户明确允许访问的，
    例如 Tailscale 使用的 `100.64.0.0/10`。这时可以通过白名单放行。
    """
    global _allowed_networks
    nets = []
    for cidr in cidrs:
        with suppress(ValueError):
            nets.append(ipaddress.ip_network(cidr, strict=False))
    _allowed_networks = nets


def _normalize_addr(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """把“映射型 IPv6 地址”规范化成真正的 IPv4 地址。

    例如 `::ffff:127.0.0.1` 实际上等价于 `127.0.0.1`。
    如果不做这一步，后面的黑名单/白名单判断可能漏判。
    """
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def _is_private(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """判断一个地址是否属于被拦截的内部/私有网络。"""
    normalized = _normalize_addr(addr)
    if _allowed_networks and any(normalized in net for net in _allowed_networks):
        return False
    return any(normalized in net for net in _BLOCKED_NETWORKS)


def validate_url_target(url: str, *, allow_loopback: bool = False) -> tuple[bool, str]:
    """校验一个 URL 是否适合被安全访问。

    【会检查什么】

    1. 协议是否是 `http/https`
    2. 是否存在主机名
    3. DNS 解析后的 IP 是否落在内网/敏感网段

    【关于 `allow_loopback`】

    这个放行非常谨慎，只允许“明确写出来的回环目标”，例如：

    - `localhost`
    - `127.0.0.1`
    - `::1`

    它不会因为某个公网域名碰巧解析到回环地址就直接放行。
    """
    try:
        p = urlparse(url)
    except Exception as e:
        return False, str(e)

    if p.scheme not in ("http", "https"):
        return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
    if not p.netloc:
        return False, "Missing domain"

    hostname = p.hostname
    if not hostname:
        return False, "Missing hostname"

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        addrs.append(addr)
    if allow_loopback and _is_allowed_loopback_target(hostname, addrs):
        return True, ""
    for addr in addrs:
        if _is_private(addr):
            return False, f"Blocked: {hostname} resolves to private/internal address {addr}"

    return True, ""


def validate_resolved_url(url: str) -> tuple[bool, str]:
    """校验一个“已解析/已跳转后的 URL”是否安全。

    这个函数常用于重定向场景：最初请求的是公网地址，但重定向后跳到了
    内网地址，这种情况同样要拦截。
    """
    try:
        p = urlparse(url)
    except Exception:
        return True, ""

    hostname = p.hostname
    if not hostname:
        return True, ""

    try:
        addr = ipaddress.ip_address(hostname)
        if _is_private(addr):
            return False, f"Redirect target is a private address: {addr}"
    except ValueError:
        # hostname is a domain name, resolve it
        try:
            infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return True, ""
        for info in infos:
            try:
                addr = ipaddress.ip_address(info[4][0])
            except ValueError:
                continue
            if _is_private(addr):
                return False, f"Redirect target {hostname} resolves to private address {addr}"

    return True, ""


def contains_internal_url(command: str, *, allow_loopback: bool = False) -> bool:
    """扫描命令字符串中是否包含指向内网/私网的 URL。"""
    for m in _URL_RE.finditer(command):
        url = m.group(0)
        ok, _ = validate_url_target(url, allow_loopback=allow_loopback)
        if not ok:
            return True
    return False


def _is_allowed_loopback_target(
    hostname: str,
    addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
) -> bool:
    """判断一个目标是否属于“可以被特殊放行的回环地址”。"""
    if not addrs or not all(_normalize_addr(addr).is_loopback for addr in addrs):
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    with suppress(ValueError):
        return ipaddress.ip_address(hostname).is_loopback
    return False
