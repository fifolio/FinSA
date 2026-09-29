from app.embeddings import embed_texts

def test_embeddings_have_expected_shape():
    vectors = embed_texts(["hello world", "another sentence"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384

def test_similar_sentences_are_closer_than_dissimilar_ones():
    vectors = embed_texts(
        [
            "Revenue declined due to weak demand.",
            "Sales dropped because demand was weak.",
            "The office is painted blue.",
        ]
    )
    v0, v1, v2 = vectors

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert dot(v0, v1) > dot(v0, v2)