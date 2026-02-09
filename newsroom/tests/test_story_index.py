import unittest

from newsroom.story_index import cluster_links, rank_clusters


class TestStoryIndex(unittest.TestCase):
    def test_cluster_links_groups_similar_stories(self) -> None:
        # Two sources about the same event, one unrelated.
        links = [
            {
                "url": "https://example.com/a",
                "norm_url": "https://example.com/a",
                "domain": "example.com",
                "title": "SpaceX to acquire xAI in massive $1.25 trillion merger",
                "description": "Elon Musk's firms plan a merger at $1.25 trillion valuation.",
                "page_age": "2026-02-03T10:00:00",
                "last_seen_ts": 1_700_000_000,
                "seen_count": 1,
            },
            {
                "url": "https://news.example.org/b",
                "norm_url": "https://news.example.org/b",
                "domain": "news.example.org",
                "title": "Musk's SpaceX merges with xAI at $1.25 trillion valuation",
                "description": "Report says SpaceX and xAI will combine in a huge deal.",
                "page_age": "2026-02-03T10:05:00",
                "last_seen_ts": 1_700_000_100,
                "seen_count": 1,
            },
            {
                "url": "https://hk.example.net/c",
                "norm_url": "https://hk.example.net/c",
                "domain": "hk.example.net",
                "title": "Hong Kong expects 1.4 million mainland visitors for Lunar New Year",
                "description": "Tourism council forecasts a 6% rise year on year.",
                "page_age": "2026-02-03T09:00:00",
                "last_seen_ts": 1_699_999_000,
                "seen_count": 1,
            },
        ]

        # now_ts should be after the published timestamps so recency scoring is meaningful.
        clusters = cluster_links(links, now_ts=1_770_116_400)
        self.assertGreaterEqual(len(clusters), 2)

        # Find the SpaceX/xAI cluster and ensure it contains 2 docs.
        sizes = sorted([len(c.docs) for c in clusters], reverse=True)
        self.assertIn(2, sizes)

    def test_cluster_links_does_not_overmerge_hong_kong_keyword(self) -> None:
        links = [
            {
                "url": "https://example.com/visitors",
                "norm_url": "https://example.com/visitors",
                "domain": "example.com",
                "title": "Hong Kong expects more visitors this week",
                "description": "Tourism forecast published today.",
                "page_age": "2026-02-03T09:00:00",
                "last_seen_ts": 1_700_000_000,
                "seen_count": 1,
            },
            {
                "url": "https://example.org/raid",
                "norm_url": "https://example.org/raid",
                "domain": "example.org",
                "title": "Hong Kong police raid office in investigation",
                "description": "Authorities say the investigation is ongoing.",
                "page_age": "2026-02-03T09:10:00",
                "last_seen_ts": 1_700_000_100,
                "seen_count": 1,
            },
        ]
        clusters = cluster_links(links, now_ts=1_770_116_400)
        # Should remain separate clusters (share only generic tokens).
        self.assertEqual(len(clusters), 2)

    def test_cluster_links_does_not_overmerge_same_outlet_locality(self) -> None:
        # Regression: two unrelated Sky News stories both mentioning Devon / West should not cluster.
        links = [
            {
                "url": "https://news.sky.com/story/london-listed-tungsten-miner-in-talks-about-16350m-share-sale-13503345",
                "norm_url": "https://news.sky.com/story/london-listed-tungsten-miner-in-talks-about-16350m-share-sale-13503345",
                "domain": "news.sky.com",
                "title": "London-listed tungsten miner in talks about £50m share sale | Money News | Sky News",
                "description": "Tungsten West, which wants to extract one of the world's largest tungsten deposits from a Devon mine, is finalising plans for a share sale which could raise more than £40m, Sky News learns.",
                "page_age": "2026-02-04T18:32:00",
                "last_seen_ts": 1_700_000_000,
                "seen_count": 1,
            },
            {
                "url": "https://news.sky.com/story/coastal-road-hit-by-three-named-storms-swept-away-into-the-sea-in-a-matter-of-hours-13502820",
                "norm_url": "https://news.sky.com/story/coastal-road-hit-by-three-named-storms-swept-away-into-the-sea-in-a-matter-of-hours-13502820",
                "domain": "news.sky.com",
                "title": "Coastal road swept away into the sea in Devon | UK News | Sky News",
                "description": "South West England was badly hit by storms Goretti, Ingrid and Chandra, which brought heavy rain and flooding last month. The A379 in Devon has been destroyed.",
                "page_age": "2026-02-03T16:27:00",
                "last_seen_ts": 1_699_999_000,
                "seen_count": 1,
            },
        ]
        clusters = cluster_links(links, now_ts=1_770_116_400)
        self.assertEqual(len(clusters), 2)

    def test_rank_clusters_prefers_newer_published_ts(self) -> None:
        links = [
            {
                "url": "https://example.com/old",
                "norm_url": "https://example.com/old",
                "domain": "example.com",
                "title": "Apple earnings report",
                "description": "Apple posts quarterly earnings and guidance.",
                "page_age": "2026-02-03T08:00:00",
                "last_seen_ts": 1_700_000_000,
                "seen_count": 1,
            },
            {
                "url": "https://example.com/new",
                "norm_url": "https://example.com/new",
                "domain": "example.com",
                "title": "UK parliament vote",
                "description": "Parliament votes on a bill after debate.",
                "page_age": "2026-02-03T10:00:00",
                "last_seen_ts": 1_700_000_100,
                "seen_count": 1,
            },
        ]
        clusters = rank_clusters(cluster_links(links, now_ts=1_770_116_400), now_ts=1_770_116_400)
        self.assertGreaterEqual(len(clusters), 2)
        self.assertEqual(clusters[0].docs[0].title, "UK parliament vote")


if __name__ == "__main__":
    unittest.main()
