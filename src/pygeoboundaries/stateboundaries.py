import numpy as np
import geopandas as gpd
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from loguru import logger
from shapely import Polygon

from pygeoboundaries.geoboundaries import get_adm

NLP_MODEL = SentenceTransformer("distiluse-base-multilingual-cased-v2")


def get_sentence_similarity(
    orig_embeding: np.ndarray, orig_sentence: str, sentence: str
) -> float:
    """
    Get the similarity score, from 0 (no similiraty) to 1 (same string),
    between two sentences

    Args:
        orig_embeding (np.ndarray): the embeding of the original sentence
        orig_sentence (str): the original sentence
        sentence (str): the sentence to compare to the original

    Returns:
        float: the similarity score, from 0 (no similiraty) to 1 (same string)
    """
    # get the similarity via the NLP model
    sentence_embedding = NLP_MODEL.encode(sentence)
    nlp_similarity = util.pytorch_cos_sim(orig_embeding, sentence_embedding)[0][
        0
    ].item()
    # get the similarity based on string comparison
    string_similarity = SequenceMatcher(None, orig_sentence, sentence).ratio()
    # get a score based on string containment and length
    contains_score = (
        0
        if orig_sentence.lower() not in sentence.lower()
        else len(orig_sentence) / len(sentence)
    )
    # combine the three scores
    return max(nlp_similarity, string_similarity, contains_score)


def get_state_boundaries(
    country: str, state: str, adm: str = "ADM1", verbose: bool = True
) -> Polygon:
    """
    Get the boundaries of a state in a country, based on how similar the
    state name is to GeoBoundaries' state names

    Args:
        country (str): the country name or ISO3 code
        state (str): the state name
        adm (str): the ADM level to get the boundaries of. Defaults to "ADM1".
        verbose (bool): whether to print a warning if the similarity score
            is low. Defaults to True.

    Returns:
        Polygon: the boundaries of the state
    """
    adm_geojsons = get_adm(country, adm)
    # convert geojson featurecollection to geopandas dataframe
    adm_gdf = gpd.GeoDataFrame.from_features(adm_geojsons.features)
    # get the embeding of the state name
    state_embedding = NLP_MODEL.encode(state)
    # get the similarity score between the state name and each state name
    # in the GeoBoundaries dataset
    adm_gdf["state_name_similarity"] = adm_gdf.shapeName.apply(
        lambda x: get_sentence_similarity(state_embedding, state, x)
    )
    # get the state with the highest similarity score
    best_match = adm_gdf.loc[adm_gdf.state_name_similarity.idxmax()]
    if best_match.state_name_similarity < 0.7 and verbose:
        logger.warning(
            "The match found could be incorrect, as the similarity score "
            f"is {best_match.state_name_similarity:.2%}. The original sentence"
            f" was '{state}', and the match was '{best_match.shapeName}'."
        )
    # return the boundaries of the best match
    return best_match.geometry
