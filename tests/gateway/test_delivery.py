"""Tests for the delivery routing module."""

from gateway.config import Platform
from gateway.delivery import DeliveryTarget
from gateway.session import SessionSource


class TestParseTargetPlatformChat:
    def test_explicit_telegram_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.is_explicit is True

    def test_platform_only_no_chat_id(self):
        target = DeliveryTarget.parse("discord")
        assert target.platform == Platform.DISCORD
        assert target.chat_id is None
        assert target.is_explicit is False

    def test_local_target(self):
        target = DeliveryTarget.parse("local")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_origin_with_source(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="789", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.thread_id == "42"
        assert target.is_origin is True

    def test_origin_without_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.platform == Platform.LOCAL
        assert target.is_origin is True

    def test_unknown_platform(self):
        target = DeliveryTarget.parse("unknown_platform")
        assert target.platform == Platform.LOCAL


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.to_string() == "origin"

    def test_local_roundtrip(self):
        target = DeliveryTarget.parse("local")
        assert target.to_string() == "local"

    def test_platform_only_roundtrip(self):
        target = DeliveryTarget.parse("discord")
        assert target.to_string() == "discord"

    def test_explicit_chat_roundtrip(self):
        target = DeliveryTarget.parse("telegram:999")
        s = target.to_string()
        assert s == "telegram:999"

        reparsed = DeliveryTarget.parse(s)
        assert reparsed.platform == Platform.TELEGRAM
        assert reparsed.chat_id == "999"


class TestCaseSensitiveChatIdParsing:
    """Test that chat IDs preserve their original case (issue #11768)."""
    
    def test_slack_uppercase_chat_id_preserved(self):
        """Slack channel IDs like C123ABC should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"  # Should NOT be lowercased to c123abc
        assert target.is_explicit is True
    
    def test_slack_chat_id_with_thread_preserved(self):
        """Slack channel:thread IDs should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABCDEF:1749236185.123456")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABCDEF"
        assert target.thread_id == "1749236185.123456"
    
    def test_matrix_room_id_preserved(self):
        """Matrix room IDs like !RoomABC:example.org stay intact."""
        target = DeliveryTarget.parse("matrix:!RoomABC:example.org")
        assert target.platform == Platform.MATRIX
        assert target.chat_id == "!RoomABC:example.org"
        assert target.thread_id is None

    def test_matrix_user_id_preserved(self):
        target = DeliveryTarget.parse("matrix:@Hermes:example.org")
        assert target.platform == Platform.MATRIX
        assert target.chat_id == "@Hermes:example.org"
        assert target.thread_id is None

    def test_matrix_alias_kept_as_single_chat_id(self):
        target = DeliveryTarget.parse("matrix:#general:example.org")
        assert target.platform == Platform.MATRIX
        assert target.chat_id == "#general:example.org"
        assert target.thread_id is None

    def test_phone_targets_preserve_e164_format(self):
        target = DeliveryTarget.parse("signal:+15551234567")
        assert target.platform == Platform.SIGNAL
        assert target.chat_id == "+15551234567"
        assert target.thread_id is None

    def test_signal_group_target_preserved(self):
        target = DeliveryTarget.parse("signal:group:abc123")
        assert target.platform == Platform.SIGNAL
        assert target.chat_id == "group:abc123"
        assert target.thread_id is None

    def test_wecom_callback_scoped_chat_id_preserved(self):
        target = DeliveryTarget.parse("wecom_callback:wwcorp123:user_a")
        assert target.platform == Platform.WECOM_CALLBACK
        assert target.chat_id == "wwcorp123:user_a"
        assert target.thread_id is None

    def test_yuanbao_direct_target_preserved(self):
        target = DeliveryTarget.parse("yuanbao:direct:acct-123")
        assert target.platform == Platform.YUANBAO
        assert target.chat_id == "direct:acct-123"
        assert target.thread_id is None

    def test_feishu_thread_target_preserved(self):
        target = DeliveryTarget.parse("feishu:oc_home:omt-thread-123")
        assert target.platform == Platform.FEISHU
        assert target.chat_id == "oc_home"
        assert target.thread_id == "omt-thread-123"

    def test_webhook_url_kept_as_single_chat_id(self):
        target = DeliveryTarget.parse("webhook:https://hooks.example.com/alerts")
        assert target.platform == Platform.WEBHOOK
        assert target.chat_id == "https://hooks.example.com/alerts"
        assert target.thread_id is None

    def test_mixed_case_chat_id_roundtrip(self):
        """Mixed-case chat IDs should survive parse-to_string roundtrip."""
        original = "telegram:ChatId123ABC"
        target = DeliveryTarget.parse(original)
        s = target.to_string()
        reparsed = DeliveryTarget.parse(s)
        assert reparsed.chat_id == "ChatId123ABC"


class TestPlatformNameCaseInsensitivity:
    """Test that platform names are case-insensitive."""
    
    def test_uppercase_platform_name(self):
        """Platform names should be case-insensitive."""
        target = DeliveryTarget.parse("TELEGRAM:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
    
    def test_mixed_case_platform_name(self):
        """Mixed-case platform names should work."""
        target = DeliveryTarget.parse("TeleGram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
