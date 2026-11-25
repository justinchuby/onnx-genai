"""Behavioral tests for the static route audit, separate from server conformance."""

import unittest

import audit_llama_server_routes as audit


class RouteAuditTests(unittest.TestCase):
    def test_methods_aliases_and_nested_layer_calls(self):
        source = """Router::new()
            .route("/one", get(routes::read).post(routes::write)
                .layer(make_layer(get(unrelated), "(ignored)")))
            .route("/two", post(routes::write))"""
        self.assertEqual(
            audit.local_routes(source),
            {
                ("GET", "/one"): "routes::read",
                ("POST", "/one"): "routes::write",
                ("POST", "/two"): "routes::write",
            },
        )

    def test_comments_are_not_registrations_and_preserve_lines(self):
        source = """// ctx_http.post("/fake", noop);
/* ctx_http.get("/also-fake", noop); */
ctx_http.get("/real", handler); // end
"""
        self.assertEqual(
            audit.upstream_routes(source),
            [{"method": "GET", "path": "/real", "source_line": 3}],
        )
        self.assertEqual(
            audit.local_routes("""// .route("/fake", get(noop))
            router.route("/real", get(handler /* comment ) */))"""),
            {("GET", "/real"): "handler"},
        )

    def test_missing_disabled_and_websocket_are_distinct(self):
        fixture = {
            "registrations": [
                {"method": "POST", "path": "/slots/:id_slot", "source_line": 1},
                {"method": "POST", "path": "/models", "source_line": 2},
                {"method": "WS", "path": "/v1/realtime", "source_line": 3},
            ]
        }
        rows = audit.contracts(
            fixture,
            """router
            .route("/slots/{id_slot}", post(routes::slots::action))
            .route("/models", post(routes::llama::disabled_feature))
            .route("/v1/realtime", get(ordinary_http))""",
        )
        self.assertEqual(
            [r["status"] for r in rows], ["present", "disabled", "missing"]
        )
        self.assertEqual(rows[0]["path"], "/slots/{id_slot}")

    def test_duplicate_upstream_gates_count_as_one_contract(self):
        fixture = {
            "registrations": [
                {"method": "GET", "path": "/health", "source_line": n} for n in [1, 2]
            ]
        }
        rows = audit.contracts(fixture, 'router.route("/health", get(discovery))')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_line"], 1)

    def test_external_harness_contracts_are_visible_but_not_missing(self):
        fixture = {"registrations": [
            {"method": method, "path": path, "source_line": 1}
            for method, path in sorted(audit.EXTERNAL_HARNESS)
        ]}
        rows = audit.contracts(fixture, 'router.route("/health", get(health))')
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["status"] == "external harness" for row in rows))
        with self.assertRaisesRegex(ValueError, "must not execute"):
            audit.contracts(fixture, 'router.route("/tools", post(exec_tools))')

    def test_explicit_websocket_registration_is_distinct_from_plain_get(self):
        fixture = {"registrations": [{"method": "WS", "path": "/v1/realtime", "source_line": 1}]}
        plain = audit.contracts(fixture, 'router.route("/v1/realtime", get(normal_http))')
        self.assertEqual(plain[0]["status"], "missing")
        upgraded = audit.contracts(fixture, 'router.route("/v1/realtime", ws(realtime::websocket::upgrade))')
        self.assertEqual(upgraded[0]["status"], "present")
        self.assertEqual(upgraded[0]["handler"], "realtime::websocket::upgrade")

    def test_local_ambiguity_and_unsupported_grammar_fail_closed(self):
        for source in [
            'router.route("/a", get(one)).route("/a", get(two))',
            'router.route("/a", some_method_router)',
            'router.route("/a", get(unclosed)',
            "no registrations here",
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                audit.local_routes(source)

    def test_upstream_methods_and_disabled_branches_remain_visible(self):
        source = """ctx_http.ws("/v1/realtime", ws);
            ctx_http.del("/models", remove);
            if (enabled) { ctx_http.get("/tools", discovery); }
            else { ctx_http.get("/tools", disabled); }"""
        rows = audit.upstream_routes(source)
        self.assertEqual([r["method"] for r in rows], ["WS", "DELETE", "GET", "GET"])
        self.assertEqual(len(rows), 4)

    def test_report_does_not_label_presence_as_conformance(self):
        fixture = {"commit": "abc", "source": "server.cpp", "registrations": [{}]}
        rows = [
            {
                "method": "GET",
                "path": "/tools",
                "source_line": 1,
                "handler": "discovery",
                "status": "present",
            }
        ]
        report = audit.render(fixture, rows)
        self.assertIn("Registration presence is not behavioral conformance", report)
        self.assertIn("Every present row still needs", report)
        self.assertIn("/abc/server.cpp#L1", report)


if __name__ == "__main__":
    unittest.main()
