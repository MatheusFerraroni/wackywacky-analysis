PAGES_COLUMNS = (
    "id",
    "domain_id",
    "parent_page_id",
    "same_as",
    "url",
    "url_md5",
    "url_final",
    "url_final_md5",
    "status_code",
    "title",
    "recursion_level",
    "status",
    "retry_count",
    "text",
    "html",
    "text_md5",
    "html_md5",
    "created_at",
    "updated_at",
)

DOMAIN_COLUMNS = (
    "id",
    "url",
    "url_md5",
    "parent_domain_id",
    "recursion_level",
    "status",
    "request_count",
    "created_at",
    "last_request_at",
)

WIKIMEDIA_HOSTS = frozenset(
    {
        "wikipedia.org",
        "wikibooks.org",
        "wikidata.org",
        "wikimedia.org",
        "wikinews.org",
        "wikiquote.org",
        "wikisource.org",
        "wikiversity.org",
        "wikivoyage.org",
        "wiktionary.org",
    }
)
