from fastapi import APIRouter
from utils.similarity import phrase_search
from utils.entity_recognition import is_location_matching

router = APIRouter()

@router.get('/search')
def search_documents(q: str):
    #raise an exception if Vectorizer is not Initialized.
    if vectorizer is None:
        raise HTTPException(status_code=400, detail="You dont have any Indexed Data, Please Refer to Indexing Documentation.")
    # query
    query = q
    print(query)
    #Entity Recognition
    ner_results = nlp(query)
    filters = {"location": None}
    for ent in ner_results:
        if ent["entity_group"] == "LOC":
            filters["location"] = ent["word"]
            break

    # Encode the query
    query_vector = encoder.encode(query)

    # Perform the nearest neighbor search
    results = index.get_nns_by_vector(query_vector, n=1000, include_distances=True)

    #phrase results
    phrase_results = phrase_search(query, vectorizer, tfidf_matrix, document_ids)

    # Retrieve the search results
    results_doc = []
    for idx, score in zip(results[0], results[1]):
        document = {
            'id': list(id_mapping.keys())[list(id_mapping.values()).index(idx)],
            'name': document_data[idx]['name'],
            'address': document_data[idx]['address'],
            'semantic_search_score': score,
            'phrase_search_score': phrase_results[idx],
            'final_score': ((score*6) + (phrase_results[idx]*4))/10
        }
        results_doc.append(document)
    if filters["location"] is not None:
        results_doc = [doc for doc in results_doc if is_location_matching(filters["location"], doc["address"])]
    response = sorted(results_doc, key= lambda d: d["final_score"], reverse=True)[0:10]
    return {"query": query, "filters": filters, "hits": response}