"""Multimodal search POC (CLIP text<->image, run locally in the app).

Unlike the sparse/dense chapters, the model (CLIP) runs in OUR process, not inside
OpenSearch: we embed the image pixels ourselves and OpenSearch is only the kNN vector
store. CLIP places images and text in one shared space, so a text query lands near the
images it describes. This chapter finally searches the image *content*, not the captions.
"""
