This is mainly for data generating

preprocess_1.py
- get labels, splits, edge_index, edge_type*, num_property_tensor, cat_property_tensor, id_tweet

preprocess_2.py
- derive tweet embeddings

preprocess_3.py
- get the neighbors(dict{idx: list}) for each user

cat.py
- concatenate des and tweet for a user