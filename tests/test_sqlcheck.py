from src.sqlcheck import execution_match, extract_sql, normalize, results_match, run

CTX = (
    "CREATE TABLE t (id INT, region TEXT, score FLOAT);"
    "INSERT INTO t VALUES (1,'EU',0.5),(2,'US',0.9),(3,'EU',0.7);"
)


def test_extract_sql_from_fence_and_prose():
    assert extract_sql("Here you go:\n```sql\nSELECT 1\n```\nHope it helps") == "SELECT 1;"
    assert extract_sql("SELECT a FROM t; SELECT b FROM t;") == "SELECT a FROM t;"
    assert extract_sql("") == ""


def test_normalize_ignores_case_space_and_semicolon():
    assert normalize("SELECT  *\nFROM t;") == normalize("select * from t")


def test_equivalent_queries_match_by_execution():
    gold = "SELECT AVG(score) FROM t WHERE region = 'EU';"
    pred = "SELECT avg(score) FROM t WHERE region IN ('EU');"
    assert execution_match(CTX, gold, pred)


def test_wrong_query_does_not_match():
    assert not execution_match(CTX, "SELECT COUNT(*) FROM t;", "SELECT COUNT(*) FROM t WHERE region='US';")


def test_invalid_prediction_is_a_miss():
    assert not execution_match(CTX, "SELECT id FROM t;", "SELEC id FROM t;")


def test_order_only_matters_with_order_by():
    assert results_match([(1,), (2,)], [(2,), (1,)], ordered=False)
    assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
    assert execution_match(CTX, "SELECT id FROM t ORDER BY score DESC;", "SELECT id FROM t ORDER BY score DESC;")
    assert not execution_match(CTX, "SELECT id FROM t ORDER BY score DESC;", "SELECT id FROM t ORDER BY score ASC;")


def test_float_rounding():
    assert results_match([(0.6000000001,)], [(0.6,)], ordered=False)


def test_runaway_query_is_stopped():
    big = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c;"
    try:
        run(CTX, big, timeout_ops=50)
        stopped = False
    except Exception:
        stopped = True
    assert stopped
