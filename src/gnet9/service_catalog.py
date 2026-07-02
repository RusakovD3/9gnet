"""Каталог серверов и прикладных сервисов уровня L0.

VLC здесь обозначает программную платформу доставки медиа, а не кодек.
Для воспроизводимости явно заданы реальные кодеки Opus и H.264/AVC.
"""

from __future__ import annotations

from enum import Enum

from .models import CodecComponent, CodecProfile, ServerProfile, ServerRuntimeMetrics, ServiceProfile


class QualityGrade(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    FAILED = "failed"


SERVER_SOURCE = "https://www.dell.com/support/manuals/en-us/poweredge-r660/pe-r660-owners-manual/technical-specifications"


SERVER_CATALOG = (
    ServerProfile("SRV_MEDIA", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6448Y", 2, 64, 512, 7.68, (25, 25), ("SVC_VOICE", "SVC_VLC"), SERVER_SOURCE),
    ServerProfile("SRV_DATA", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6430", 2, 64, 256, 15.36, (25, 25), ("SVC_FTP", "SVC_DNS"), SERVER_SOURCE),
    ServerProfile("SRV_RTC", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6448Y", 2, 64, 512, 7.68, (25, 25), ("SVC_TELEMOST",), SERVER_SOURCE),
    ServerProfile("SRV_LIVE", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6448Y", 2, 64, 512, 15.36, (25, 25), ("SVC_LIVE",), SERVER_SOURCE),
)

SERVER_BASELINE_RUNTIME = {
    "SRV_MEDIA": ServerRuntimeMetrics(18.0, 32.0, 8.0, 28.0, 43.0, 80, 0.995),
    "SRV_DATA": ServerRuntimeMetrics(14.0, 38.0, 6.0, 42.0, 41.0, 80, 0.996),
    "SRV_RTC": ServerRuntimeMetrics(22.0, 34.0, 11.0, 24.0, 45.0, 40, 0.994),
    "SRV_LIVE": ServerRuntimeMetrics(20.0, 31.0, 13.0, 36.0, 44.0, 40, 0.994),
}

CODEC_CATALOG = {
    "voice_opus_rtp": CodecProfile(
        profile_id="voice_opus_rtp",
        display_name="Opus fullband mono по RTP/UDP",
        transport_stack="Ethernet / IPv4 / UDP / RTP",
        application_protocol="RTP audio stream",
        components=(
            CodecComponent(
                media="audio",
                codec="Opus",
                profile="fullband speech, mono",
                payload_or_container="RTP dynamic payload type 111",
                bitrate_note="16/24/32 кбит/с по SLA; 20 мс пакетизация",
                clock_rate_hz=48_000,
                channels=1,
                packetization_ms=20,
            ),
        ),
        realism_note="Реалистичный профиль интерактивной речи: Opus, 48 кГц RTP clock, 20 мс кадр.",
    ),
    "vlc_opus_h264_rtp": CodecProfile(
        profile_id="vlc_opus_h264_rtp",
        display_name="VLC: Opus audio + H.264/AVC video по RTP/UDP",
        transport_stack="Ethernet / IPv4 / UDP / RTP",
        application_protocol="VLC RTP media stream",
        components=(
            CodecComponent(
                media="audio",
                codec="Opus",
                profile="fullband audio",
                payload_or_container="RTP dynamic payload",
                bitrate_note="64/128/320 кбит/с по d0sl",
                clock_rate_hz=48_000,
                channels=2,
                packetization_ms=20,
            ),
            CodecComponent(
                media="video",
                codec="H.264/AVC",
                profile="Main/High profile, progressive 30 fps",
                payload_or_container="RTP payload for H.264 NAL units",
                bitrate_note="1000/2000/4000 кбит/с по d0sl",
                clock_rate_hz=90_000,
                frame_rate_fps=30.0,
            ),
        ),
        realism_note="VLC — платформа доставки, не кодек; аудио и видео явно разведены на Opus и H.264/AVC.",
    ),
    "ftp_tcp_binary": CodecProfile(
        profile_id="ftp_tcp_binary",
        display_name="FTP: бинарная полезная нагрузка по TCP",
        transport_stack="Ethernet / IPv4 / TCP",
        application_protocol="FTP data connection",
        components=(
            CodecComponent(
                media="file",
                codec="binary/octet-stream",
                profile="файловые данные без медиакодека",
                payload_or_container="TCP byte stream",
                bitrate_note="512/2048/4096 кбит/с полезной нагрузки по d0sl",
            ),
        ),
        realism_note="Для FTP термин «кодек» неприменим; профиль описывает протокол и формат полезной нагрузки.",
    ),
    "dns_udp_messages": CodecProfile(
        profile_id="dns_udp_messages",
        display_name="DNS: сообщения запрос/ответ по UDP с TCP fallback",
        transport_stack="Ethernet / IPv4 / UDP; TCP fallback для крупных ответов",
        application_protocol="DNS",
        components=(
            CodecComponent(
                media="control",
                codec="DNS wire format",
                profile="query/response resource records",
                payload_or_container="UDP DNS message",
                bitrate_note="16/32/64 кбит/с эквивалентной нагрузки модели",
            ),
        ),
        realism_note="DNS не использует медиакодек; модель описывает wire-format сообщений и основной UDP transport.",
    ),
    "webrtc_opus_h264": CodecProfile(
        profile_id="webrtc_opus_h264",
        display_name="WebRTC: Opus audio + H.264/AVC video",
        transport_stack="Ethernet / IPv4 / UDP / RTP; SRTP/DTLS не шифруется в модели",
        application_protocol="WebRTC-compatible RTP media",
        components=(
            CodecComponent(
                media="audio",
                codec="Opus",
                profile="interactive audio",
                payload_or_container="RTP payload",
                bitrate_note="аудио оценивается отдельным бюджетом задержки",
                clock_rate_hz=48_000,
                channels=1,
                packetization_ms=20,
            ),
            CodecComponent(
                media="video",
                codec="H.264/AVC",
                profile="WebRTC-compatible video, VP8 fallback допустим",
                payload_or_container="RTP payload",
                bitrate_note="1024/2048/4096 кбит/с по d0sl",
                clock_rate_hz=90_000,
                frame_rate_fps=30.0,
            ),
        ),
        realism_note="Профиль отражает типичный WebRTC-набор; криптографический слой SRTP/DTLS намеренно не моделируется.",
    ),
    "ll_hls_aac_h264": CodecProfile(
        profile_id="ll_hls_aac_h264",
        display_name="Low-Latency HLS: AAC-LC audio + H.264/AVC video",
        transport_stack="Ethernet / IPv4 / TCP / TLS / HTTP",
        application_protocol="Low-Latency HLS",
        components=(
            CodecComponent(
                media="audio",
                codec="AAC-LC",
                profile="stereo audio",
                payload_or_container="fMP4/CMAF segment",
                bitrate_note="служебная часть профиля live streaming",
                clock_rate_hz=48_000,
                channels=2,
            ),
            CodecComponent(
                media="video",
                codec="H.264/AVC",
                profile="Main/High profile",
                payload_or_container="fMP4/CMAF segment over HTTP",
                bitrate_note="2000/3500/6000 кбит/с по d0sl",
                frame_rate_fps=30.0,
            ),
        ),
        realism_note="Для HLS задержка задаётся секундами, а не миллисекундами; транспорт идёт поверх HTTP/TCP.",
    ),
}

TRAFFIC_CODEC_PROFILE = {
    "voice": "voice_opus_rtp",
    "broadcast_mp3": "vlc_opus_h264_rtp",
    "vlc_av": "vlc_opus_h264_rtp",
    "ftp": "ftp_tcp_binary",
    "dns": "dns_udp_messages",
    "video_conference": "webrtc_opus_h264",
    "live_streaming": "ll_hls_aac_h264",
}


SERVICE_CATALOG = (
    ServiceProfile("Voice", 0.032, 80.0, 20.0, 0.9995, "gold", "SVC_VOICE", "SRV_MEDIA", "interactive-audio", "RTP", "Opus", None, None, "voice_opus_rtp"),
    ServiceProfile("VLC AV", 4.320, 150.0, 30.0, 0.9990, "gold", "SVC_VLC", "SRV_MEDIA", "audio-video", "VLC RTP", "Opus", "H.264/AVC", None, "vlc_opus_h264_rtp"),
    ServiceProfile("FTP", 4.096, 300.0, 120.0, 0.9950, "silver", "SVC_FTP", "SRV_DATA", "file-transfer", "FTP/TCP", None, None, None, "ftp_tcp_binary"),
    ServiceProfile("DNS", 0.064, 30.0, 10.0, 0.9999, "gold", "SVC_DNS", "SRV_DATA", "name-resolution", "DNS/UDP", None, None, None, "dns_udp_messages"),
    ServiceProfile("Telemost", 4.096, 70.0, 30.0, 0.9990, "gold", "SVC_TELEMOST", "SRV_RTC", "video-conference", "WebRTC-compatible RTP", "Opus", "H.264/AVC", 600.0, "webrtc_opus_h264"),
    ServiceProfile("Live Streaming", 6.0, 2000.0, 250.0, 0.9950, "gold", "SVC_LIVE", "SRV_LIVE", "live-streaming", "Low-Latency HLS", "AAC-LC", "H.264/AVC", 10000.0, "ll_hls_aac_h264"),
)


def codec_summary(profile_id: str) -> str:
    """Return a compact Russian summary for UI panels and exports."""
    profile = CODEC_CATALOG[profile_id]
    components = " + ".join(component.codec for component in profile.components)
    return f"{components}; {profile.transport_stack}"


def classify_tcp_retransmission(retransmission_percent: float) -> QualityGrade:
    """Оценить FTP по доле повторно переданных TCP-сегментов.

    Значения 40–80% означают отказ, поэтому рабочие границы намеренно строже.
    """
    if retransmission_percent < 0:
        raise ValueError("Доля повторных передач не может быть отрицательной")
    if retransmission_percent <= 0.1:
        return QualityGrade.GOLD
    if retransmission_percent <= 0.5:
        return QualityGrade.SILVER
    if retransmission_percent <= 1.0:
        return QualityGrade.BRONZE
    return QualityGrade.FAILED


def classify_latency(latency_ms: float, *, gold_ms: float, silver_ms: float, bronze_ms: float, failed_ms: float) -> QualityGrade:
    """Единая монотонная классификация задержки без пересекающихся диапазонов."""
    if latency_ms < 0 or not gold_ms <= silver_ms <= bronze_ms <= failed_ms:
        raise ValueError("Некорректные границы задержки")
    if latency_ms <= gold_ms:
        return QualityGrade.GOLD
    if latency_ms <= silver_ms:
        return QualityGrade.SILVER
    if latency_ms <= bronze_ms:
        return QualityGrade.BRONZE
    return QualityGrade.FAILED
