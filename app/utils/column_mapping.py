import difflib
import unicodedata

FIELD_SYNONYMS = {
    "designation": ["designation", "désignation", "article", "libellé", "description", "item"],
    "unit": ["unit", "unité", "u", "unite"],
    "pu": ["pu", "prix unitaire", "prix", "unit price", "p.u."],
    "lot": ["lot", "section", "groupe", "group"],
}

def normalize(text):
    text = str(text).strip().lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def auto_map_fields(articles):
    if not articles:
        return []
    keys = list(articles[0].keys())
    mapping = {}
    norm_keys = [normalize(k) for k in keys]
    for field, synonyms in FIELD_SYNONYMS.items():
        found = None
        for syn in synonyms:
            matches = difflib.get_close_matches(normalize(syn), norm_keys, n=1, cutoff=0.7)
            if matches:
                found = keys[norm_keys.index(matches[0])]
                break
        mapping[field] = found
    new_articles = []
    for art in articles:
        new_art = {f: art.get(mapping[f], "") for f in FIELD_SYNONYMS}
        new_articles.append(new_art)
    return new_articles