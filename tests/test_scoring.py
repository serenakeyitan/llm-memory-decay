from datetime import datetime

from llm_memory_decay.scoring import ImportanceScorer


def test_record_and_frequency():
    scorer = ImportanceScorer()
    scorer.record_mention("python")
    scorer.record_mention("python")
    scorer.record_mention("python")
    topics = {t.topic: t for t in scorer.all_topics()}
    assert topics["python"].frequency == 3


def test_frequency_weight_scales():
    scorer = ImportanceScorer()
    for _ in range(10):
        scorer.record_mention("popular")
    scorer.record_mention("rare")

    pop_weight = scorer.get_frequency_weight("popular")
    rare_weight = scorer.get_frequency_weight("rare")
    assert pop_weight > rare_weight


def test_pin_topic():
    scorer = ImportanceScorer()
    scorer.pin_topic("core_interest")
    assert scorer.is_pinned("core_interest")
    assert not scorer.is_pinned("random_topic")


def test_unknown_topic_frequency_weight():
    scorer = ImportanceScorer()
    weight = scorer.get_frequency_weight("unknown")
    assert weight == 0.1


def test_unknown_topic_importance():
    scorer = ImportanceScorer()
    importance = scorer.get_importance("unknown")
    assert importance == 1.0


def test_importance_can_be_set():
    scorer = ImportanceScorer()
    scorer.record_mention("topic", importance=2.5)
    assert scorer.get_importance("topic") == 2.5


def test_all_topics_returns_list():
    scorer = ImportanceScorer()
    scorer.record_mention("a")
    scorer.record_mention("b")
    topics = scorer.all_topics()
    assert len(topics) == 2
    topic_names = {t.topic for t in topics}
    assert "a" in topic_names
    assert "b" in topic_names


def test_last_seen_updated():
    scorer = ImportanceScorer()
    t1 = datetime(2025, 1, 1)
    t2 = datetime(2025, 6, 1)
    scorer.record_mention("topic", at=t1)
    scorer.record_mention("topic", at=t2)
    topics = {t.topic: t for t in scorer.all_topics()}
    assert topics["topic"].last_seen == t2
