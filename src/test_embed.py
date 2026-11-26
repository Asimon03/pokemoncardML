from embed import embed_images

paths = ["test_images/testimg.png", "test_images/testimg_copy.png", "test_images/testimg_dif.png"]
embs = embed_images(paths)

print("embedding shapes:", embs.shape)
print("embeddings\n-------------------")
i = 1
for emb in embs:
    print("\n\nembedding " + str(i))
    print(emb)
    i += 1