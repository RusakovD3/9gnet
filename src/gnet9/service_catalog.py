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


SERVER_SOURCE = (
    "https://www.dell.com/support/manuals/en-us/poweredge-r660/"
    "r660_ism_pub/processor-specifications?guid=guid-3518d4c6-08f3-4817-8906-c6f9358a00c7&lang=en-us"
)
SERVER_NIC_SOURCE = (
    "https://www.dell.com/support/manuals/en-au/oth-r660/"
    "r660_ism_pub/nic-port-specifications?guid=guid-29a0704e-235f-43bb-a8a6-25fbc60ca3b9&lang=en-us"
)
SERVER_MEMORY_SOURCE = (
    "https://www.dell.com/support/manuals/en-us/oth-r660/"
    "r660_ism_pub/memory-specifications?guid=guid-e397ae52-25e2-4a69-a93d-232d818a35ad"
)
SERVER_POWER_SOURCE = (
    "https://www.dell.com/support/manuals/en-us/poweredge-r660/"
    "r660_ism_pub/psu-specifications?guid=guid-e770480f-00b7-476a-ab26-5690ef0e8601&lang=en-us"
)
INTEL_6430_SOURCE = (
    "https://www.intel.com/content/www/us/en/products/sku/231737/"
    "intel-xeon-gold-6430-processor-60m-cache-2-10-ghz/specifications.html"
)
INTEL_6448Y_SOURCE = (
    "https://www.intel.com/content/www/us/en/products/sku/232384/"
    "intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz/specifications.html"
)

RFC_OPUS = "https://www.rfc-editor.org/rfc/rfc7587.html"
RFC_H264_RTP = "https://www.rfc-editor.org/rfc/rfc6184.html"
RFC_WEBRTC_VIDEO = "https://www.rfc-editor.org/rfc/rfc7742.html"
RFC_WEBRTC_MEDIA = "https://www.rfc-editor.org/rfc/rfc8834.html"
RFC_FTP = "https://www.rfc-editor.org/rfc/rfc959.html"
RFC_DNS = "https://www.rfc-editor.org/rfc/rfc1035.html"
RFC_DNS_TCP = "https://www.rfc-editor.org/rfc/rfc7766.html"
ITU_G114 = "https://www.itu.int/rec/T-REC-G.114"
W3C_WEBRTC = "https://www.w3.org/TR/webrtc/"
APPLE_LL_HLS = (
    "https://developer.apple.com/documentation/http-live-streaming/"
    "hls-authoring-specification-for-apple-devices"
)


SERVER_CATALOG = (
    ServerProfile(
        "SRV_MEDIA", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6448Y",
        2, 64, 512, 7.68, (10, 10), ("SVC_VOICE", "SVC_VLC"), SERVER_SOURCE,
        (SERVER_SOURCE, SERVER_NIC_SOURCE, SERVER_MEMORY_SOURCE, SERVER_POWER_SOURCE, INTEL_6448Y_SOURCE),
        psu_source_url=SERVER_POWER_SOURCE,
    ),
    ServerProfile(
        "SRV_DATA", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6430",
        2, 64, 256, 15.36, (10, 10), ("SVC_FTP", "SVC_DNS"), SERVER_SOURCE,
        (SERVER_SOURCE, SERVER_NIC_SOURCE, SERVER_MEMORY_SOURCE, SERVER_POWER_SOURCE, INTEL_6430_SOURCE),
        psu_source_url=SERVER_POWER_SOURCE,
    ),
    ServerProfile(
        "SRV_RTC", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6448Y",
        2, 64, 512, 7.68, (10, 10), ("SVC_TELEMOST",), SERVER_SOURCE,
        (SERVER_SOURCE, SERVER_NIC_SOURCE, SERVER_MEMORY_SOURCE, SERVER_POWER_SOURCE, INTEL_6448Y_SOURCE),
        psu_source_url=SERVER_POWER_SOURCE,
    ),
    ServerProfile(
        "SRV_LIVE", "Dell PowerEdge R660", "2 × Intel Xeon Gold 6448Y",
        2, 64, 512, 15.36, (10, 10), ("SVC_LIVE",), SERVER_SOURCE,
        (SERVER_SOURCE, SERVER_NIC_SOURCE, SERVER_MEMORY_SOURCE, SERVER_POWER_SOURCE, INTEL_6448Y_SOURCE),
        psu_source_url=SERVER_POWER_SOURCE,
    ),
)

SERVER_BASELINE_RUNTIME = {
    # Сетевая доля относится к сумме выбранных 2×10GbE и включает небольшой
    # запас на репликацию/управление сверх 480 явно моделируемых потоков.
    "SRV_MEDIA": ServerRuntimeMetrics(18.0, 32.0, 2.2, 28.0, 43.0, 160, 0.995),
    "SRV_DATA": ServerRuntimeMetrics(14.0, 38.0, 2.1, 42.0, 41.0, 160, 0.996),
    "SRV_RTC": ServerRuntimeMetrics(22.0, 34.0, 2.0, 24.0, 45.0, 80, 0.994),
    "SRV_LIVE": ServerRuntimeMetrics(20.0, 31.0, 3.0, 36.0, 44.0, 80, 0.994),
}

CODEC_CATALOG = {
    "voice_opus_rtp": CodecProfile(
        profile_id="voice_opus_rtp",
        display_name="Opus wideband/fullband mono по RTP/UDP",
        transport_stack="Ethernet / IPv4 / UDP / RTP",
        application_protocol="RTP audio stream",
        components=(
            CodecComponent(
                media="audio",
                codec="Opus",
                profile="adaptive wideband/fullband speech, mono",
                payload_or_container="RTP dynamic payload type 111",
                bitrate_note="16/24/32 кбит/с по SLA; 20 мс пакетизация",
                clock_rate_hz=48_000,
                channels=1,
                packetization_ms=20,
            ),
        ),
        realism_note=(
            "Opus использует RTP clock 48 кГц и кадр 20 мс; 16 кбит/с относится "
            "к wideband, а 32 кбит/с попадает в рекомендованный fullband speech диапазон."
        ),
        reference_urls=(RFC_OPUS, ITU_G114),
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
        realism_note=(
            "VLC — платформа доставки, не кодек; 320 кбит/с для Opus допустимы "
            "диапазоном RFC, но являются повышенным SLA-профилем, а не sweet spot речи."
        ),
        reference_urls=(RFC_OPUS, RFC_H264_RTP),
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
        reference_urls=(RFC_FTP,),
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
        realism_note=(
            "DNS не использует медиакодек; модель описывает wire-format, UDP и "
            "обязательную поддержку TCP для полноценной реализации."
        ),
        reference_urls=(RFC_DNS, RFC_DNS_TCP),
    ),
    "webrtc_opus_h264": CodecProfile(
        profile_id="webrtc_opus_h264",
        display_name="WebRTC: Opus + VP8/H.264 Constrained Baseline",
        transport_stack="Ethernet / IPv4 / UDP / SRTP/SRTCP; DTLS-SRTP",
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
                codec="VP8 или H.264/AVC",
                profile="VP8 либо H.264 Constrained Baseline для базовой совместимости WebRTC",
                payload_or_container="RTP payload",
                bitrate_note="1024/2048/4096 кбит/с по d0sl",
                clock_rate_hz=90_000,
                frame_rate_fps=30.0,
            ),
        ),
        realism_note=(
            "Профиль следует базовой совместимости WebRTC. Пакетная модель считает RTP/UDP; "
            "дополнительные байты SRTP/SRTCP и DTLS-handshake пока не включены и явно "
            "помечаются в потоке как ограничение симуляции."
        ),
        reference_urls=(W3C_WEBRTC, RFC_OPUS, RFC_WEBRTC_VIDEO, RFC_WEBRTC_MEDIA, ITU_G114),
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
        realism_note=(
            "Для HLS задержка задаётся секундами, а не миллисекундами. Gold 2 с — "
            "агрессивная инженерная цель; Apple рекомендует part target 1 с и "
            "PART-HOLD-BACK не менее трёх part target."
        ),
        reference_urls=(APPLE_LL_HLS,),
    ),
}

TRAFFIC_CODEC_PROFILE = {
    "voice": "voice_opus_rtp",
    "vlc_av": "vlc_opus_h264_rtp",
    "ftp": "ftp_tcp_binary",
    "dns": "dns_udp_messages",
    "video_conference": "webrtc_opus_h264",
    "live_streaming": "ll_hls_aac_h264",
}


SERVICE_CATALOG = (
    ServiceProfile(
        name="Голос", bitrate_mbps=0.032, latency_ms_max=80.0,
        jitter_ms_max=20.0, availability_target=0.9995, priority="gold",
        service_id="SVC_VOICE", server_id="SRV_MEDIA",
        category="interactive-audio", platform="RTP", audio_codec="Opus",
        codec_profile_id="voice_opus_rtp", reference_urls=(RFC_OPUS, ITU_G114),
        primary_slo_metrics=("one_way_mouth_to_ear_latency", "network_packet_loss", "jitter"),
    ),
    ServiceProfile(
        name="VLC: аудио и видео", bitrate_mbps=4.320, latency_ms_max=150.0,
        jitter_ms_max=30.0, availability_target=0.9990, priority="gold",
        service_id="SVC_VLC", server_id="SRV_MEDIA", category="audio-video",
        platform="VLC RTP", audio_codec="Opus", video_codec="H.264/AVC",
        codec_profile_id="vlc_opus_h264_rtp",
        reference_urls=(RFC_OPUS, RFC_H264_RTP),
        primary_slo_metrics=("one_way_media_latency", "network_packet_loss", "media_bitrate"),
    ),
    ServiceProfile(
        name="Передача файлов (FTP)", bitrate_mbps=4.096, latency_ms_max=300.0,
        jitter_ms_max=120.0, availability_target=0.9950, priority="silver",
        service_id="SVC_FTP", server_id="SRV_DATA", category="file-transfer",
        platform="FTP/TCP", codec_profile_id="ftp_tcp_binary",
        reference_urls=(RFC_FTP,),
        primary_slo_metrics=("goodput", "file_completion_time", "transfer_success", "tcp_retransmission_ratio"),
        modeling_limitations=("latency_and_jitter_fields_are_compatibility_guardrails_not_primary_ftp_slo",),
    ),
    ServiceProfile(
        name="Разрешение имён (DNS)", bitrate_mbps=0.064, latency_ms_max=50.0,
        jitter_ms_max=10.0, availability_target=0.9999, priority="gold",
        service_id="SVC_DNS", server_id="SRV_DATA", category="name-resolution",
        platform="DNS/UDP + TCP fallback", codec_profile_id="dns_udp_messages",
        reference_urls=(RFC_DNS, RFC_DNS_TCP),
        primary_slo_metrics=("response_time_p95_p99", "timeout_ratio", "servfail_ratio", "tcp_fallback_success"),
        modeling_limitations=(
            "bitrate_is_equivalent_load_for_generator_not_a_dns_service_slo",
            "gold_dns_50ms_is_a_gnet9_operator_p95_target_not_an_rfc_requirement",
        ),
    ),
    ServiceProfile(
        name="Видеоконференция", bitrate_mbps=4.096, latency_ms_max=70.0,
        jitter_ms_max=30.0, availability_target=0.9990, priority="gold",
        service_id="SVC_TELEMOST", server_id="SRV_RTC",
        category="video-conference", platform="WebRTC media profile (DTLS-SRTP)",
        audio_codec="Opus", video_codec="VP8 или H.264 Constrained Baseline", critical_latency_ms=600.0,
        codec_profile_id="webrtc_opus_h264",
        reference_urls=(W3C_WEBRTC, RFC_OPUS, RFC_WEBRTC_VIDEO, RFC_WEBRTC_MEDIA, ITU_G114),
        primary_slo_metrics=("audio_one_way_latency", "video_one_way_latency", "media_loss", "jitter"),
        modeling_limitations=("srtp_srtcp_and_dtls_byte_overhead_not_yet_counted",),
    ),
    ServiceProfile(
        name="Прямая трансляция", bitrate_mbps=6.0, latency_ms_max=2000.0,
        jitter_ms_max=250.0, availability_target=0.9950, priority="gold",
        service_id="SVC_LIVE", server_id="SRV_LIVE", category="live-streaming",
        platform="Low-Latency HLS", audio_codec="AAC-LC", video_codec="H.264/AVC",
        critical_latency_ms=10000.0, codec_profile_id="ll_hls_aac_h264",
        reference_urls=(APPLE_LL_HLS,),
        primary_slo_metrics=("live_edge_latency", "startup_time", "rebuffer_ratio", "part_deadline_miss_ratio"),
        modeling_limitations=("jitter_field_is_a_transport_guardrail_not_primary_ll_hls_user_slo",),
    ),
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
