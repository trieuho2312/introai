from recommender.recommend import SongRecommender

def test_recommend():
    recommender = SongRecommender()
    result = recommender.recommend("Shape of You")
    assert result is not None
    assert not result.empty