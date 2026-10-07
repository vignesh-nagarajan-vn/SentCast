import datetime as dt

from sentcast import corpus
from sentcast.config import load_config


def test_headline_from_url_keeps_words_and_drops_ids():
    assert corpus.headline_from_url(
        "https://www.cnbc.com/2018/06/01/stocks-jump-after-strong-jobs-report.html"
    ) == "stocks jump after strong jobs report"
    assert corpus.headline_from_url(
        "https://www.reuters.com/article/us-usa-fed/fed-holds-rates-steady-signals-two-cuts-in-2024-idUSKBN2XYZ12"
    ) == "fed holds rates steady signals two cuts in 2024"


def test_headline_from_url_rejects_urls_without_words():
    assert corpus.headline_from_url("https://www.tmcnet.com/usubmit/2021/06/01/9380994.htm") == ""
    assert corpus.headline_from_url(
        "http://www.richmond.com/ap/article_5e490322-19f2-5a6a-ad53-ba0b67fa0460.html") == ""
    assert corpus.headline_from_url("not a url") == ""


def test_clean_title_strips_entities_and_site_name():
    assert corpus.clean_title("Stocks &amp; bonds slump as rates rise | News Radio 101.9") == \
        "Stocks & bonds slump as rates rise"
    assert corpus.clean_title("Short | Site") == "Short | Site"


def test_slot_draw_is_seeded_and_without_replacement():
    day = dt.date(2020, 3, 16)
    a = corpus.slots_for_day(day, 1)
    assert a == corpus.slots_for_day(day, 1)
    assert a != corpus.slots_for_day(day, 2)
    assert a != corpus.slots_for_day(dt.date(2020, 3, 17), 1)
    assert sorted(a) == list(range(96))


def test_slot_timestamp():
    assert corpus.slot_timestamp(dt.date(2020, 3, 16), 0) == "20200316000000"
    assert corpus.slot_timestamp(dt.date(2020, 3, 16), 95) == "20200316234500"


def _row(url, locations="1#United States#US#US#38#-97#US", collection="1", domain="example.com", extras=""):
    f = [""] * 27
    f[0], f[1], f[2], f[3], f[4] = "20200316120000-1", "20200316120000", collection, domain, url
    f[9] = locations
    f[15] = "-2.5,1.0,3.5,4.5,20.0,1.0,300"
    f[26] = extras
    return "\t".join(f)


def test_parse_gkg_applies_each_filter():
    cfg = load_config()["gkg"]
    lines = [
        _row("https://a.com/x/stocks-fall-as-inflation-fears-grow", extras="<PAGE_TITLE>Stocks fall | A</PAGE_TITLE>"),
        _row("https://a.com/x/stocks-fall-as-inflation-fears-grow", locations="1#France#FR#FR#46#2#FR"),   # not US
        _row("https://a.com/x/local-team-wins-the-big-game"),                                              # no keyword
        _row("https://a.com/x/1234.html"),                                                                 # no headline
        _row("https://prnewswire.com/x/stocks-rally-on-new-product-launch", domain="prnewswire.com"),      # press release
        _row("BBC Monitoring/(c) BBC", collection="2"),                                                    # not a web article
    ]
    rows, funnel = corpus.parse_gkg("\n".join(lines).encode(), cfg)
    assert funnel == {"rows": 6, "web": 5, "us": 4, "headline": 3, "keyword": 2, "kept": 1}
    assert rows[0]["headline"] == "stocks fall as inflation fears grow"
    assert rows[0]["tone"] == -2.5 and rows[0]["wordcount"] == 300
    assert rows[0]["title"] == "Stocks fall | A"
