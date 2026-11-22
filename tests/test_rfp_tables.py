from __future__ import annotations

import unittest

from rfp_tables import (
    TableData,
    analyse_table,
    boq_requirements,
    column_roles,
    is_piece_unit,
    parse_quantity,
    row_record,
    table_markdown,
)


def table(header, rows, **kw) -> TableData:
    return analyse_table(TableData(id="T1", header=header, rows=rows, pages=[2],
                                   row_pages=[2] * len(rows), section="2. Scope", **kw))


class QuantityTests(unittest.TestCase):
    def test_numbers_units_and_indian_grouping(self) -> None:
        self.assertEqual(parse_quantity("1,200"), (1200, ""))
        self.assertEqual(parse_quantity("1,20,000"), (120000, ""))
        self.assertEqual(parse_quantity("120 Nos"), (120, "Nos"))
        self.assertEqual(parse_quantity("2 Sets"), (2, "Sets"))
        self.assertEqual(parse_quantity("approx. 50"), (50, ""))
        self.assertEqual(parse_quantity("40.0"), (40, ""))

    def test_not_a_quantity(self) -> None:
        for text in ("", "-", "As required", "RJ-45", "0", "12.5 m"):
            with self.subTest(text=text):
                self.assertIsNone(parse_quantity(text)[0])
        self.assertEqual(parse_quantity("12.5 m"), (None, "m"))

    def test_piece_units(self) -> None:
        for unit in ("Nos", "No.", "pcs", "Each", "Set", ""):
            self.assertTrue(is_piece_unit(unit), unit)
        for unit in ("Box", "Mtr", "Coil", "Kg"):
            self.assertFalse(is_piece_unit(unit), unit)


class RoleTests(unittest.TestCase):
    def test_common_boq_headers(self) -> None:
        roles = column_roles(["Sl. No.", "Description of Item", "Make", "UoM", "Qty", "Unit Rate (Rs.)", "Amount"],
                             [["1", "Switch", "Havells", "Nos", "10", "", ""]])
        self.assertEqual(roles, ["sno", "description", "make", "unit", "quantity", "price", "price"])

    def test_wrapped_short_headers(self) -> None:
        self.assertEqual(column_roles(["S.N o", "Item", "Qty"], [["1", "Switch", "5"]])[0], "sno")

    def test_location_columns_are_parts_and_total_is_the_quantity(self) -> None:
        roles = column_roles(["Item", "Unit", "Block A", "Block B", "Total Qty"],
                             [["Plate", "Nos", "5", "7", "12"], ["Box", "Nos", "1", "2", "3"]])
        self.assertEqual(roles, ["description", "unit", "quantity_part", "quantity_part", "quantity"])

    def test_two_quantity_columns_without_total_are_both_parts(self) -> None:
        roles = column_roles(["Item", "Qty Phase 1", "Qty Phase 2"], [["Plate", "5", "7"]])
        self.assertEqual(roles, ["description", "quantity_part", "quantity_part"])

    def test_unnamed_text_column_becomes_the_description(self) -> None:
        roles = column_roles(["#", "Name", "Qty"], [["1", "16A switch with indicator", "4"]])
        self.assertEqual(roles[1], "description")

    def test_specification_column_is_the_description_when_there_is_none(self) -> None:
        roles = column_roles(["S.No", "Technical Specification", "Qty"], [["1", "6A 1 way switch, white", "40"]])
        self.assertEqual(roles, ["sno", "description", "quantity"])


class ClassifyTests(unittest.TestCase):
    def test_kinds(self) -> None:
        self.assertEqual(table(["Item", "Qty"], [["Switch", "10"]]).kind, "boq")
        self.assertEqual(table(["Tender No", "ABC/2026/1"], [["EMD", "Rs 10,000"]]).kind, "kv")
        self.assertEqual(table(["Parameter", "Requirement"], [["Voltage", "240 V"]]).kind, "spec")
        self.assertEqual(table(["Lot", "Floors", "Delivery within"], [["1", "3-4", "10 days"]]).kind, "other")

    def test_gem_consignee_table_is_not_an_item_list(self) -> None:
        # Headers as PyMuPDF reads a real GeM bid (Hindi part garbled by the font)
        t = table(["5.सं./S.N o.", "परे\x01षती/Pरपो;टdग अिधकार\x13 /Consignee Reporting/Officer", "पता/Address",
                   "मा ा /Quantity", ";डलीवर\x13 के ;दन/Delivery Days"],
                  [["1", "Lavdeep Kumar", "140124,Bhakra Beas Management Board, Irrigation Wing, NANGAL TOWNSHIP",
                    "100", "15"]])
        self.assertEqual(t.roles[1:], ["location", "location", "quantity", "delivery"])
        self.assertEqual(t.kind, "consignee")

    def test_consignee_column_inside_an_item_list_is_an_attribute(self) -> None:
        t = table(["Item", "Qty", "Consignee"], [["16A switch", "40", "Stores, Nangal"]])
        self.assertEqual(t.kind, "boq")
        r = boq_requirements(t)[0]
        self.assertEqual((r.requirement, r.quantity, r.attributes), ("16A switch", 40, {"Consignee": "Stores, Nangal"}))

    def test_item_table_without_any_quantity_is_not_boq(self) -> None:
        self.assertEqual(table(["S.No", "Item", "Qty"], [["1", "Switch", "As required"]]).kind, "other")


class BoqRequirementTests(unittest.TestCase):
    def test_category_rows_totals_specs_and_attributes(self) -> None:
        t = table(
            ["S.No", "Description", "Specification", "Make", "Unit", "Qty", "Rate"],
            [
                ["", "A. Switches", "", "", "", "", ""],
                ["1", "1 way switch", "6A, white", "Havells", "Nos", "1,200", ""],
                ["2", "Bell push switch", "", "", "Nos", "40", ""],
                ["", "B. Sockets", "", "", "", "", ""],
                ["3", "3 pin socket", "", "Havells", "Nos", "300", ""],
                ["", "Total", "", "", "", "", ""],
            ],
        )
        reqs = boq_requirements(t)
        # "Switches" / "Sockets" add nothing to the descriptions, so they are not prefixed
        self.assertEqual([r.requirement for r in reqs],
                         ["1 way switch, 6A, white", "Bell push switch", "3 pin socket"])
        self.assertEqual([r.category for r in reqs], ["Switches", "Switches", "Sockets"])
        self.assertEqual([r.quantity for r in reqs], [1200, 40, 300])
        self.assertEqual([r.row_label for r in reqs], ["1", "2", "3"])
        self.assertEqual(reqs[0].attributes, {"Make": "Havells"})
        self.assertEqual(reqs[0].page, 2)

    def test_category_is_added_only_when_it_says_more(self) -> None:
        t = table(["Category", "Item", "Qty"], [["Switches", "16A switch", "5"],
                                                ["Plates (Coral Magna, Salmon Red)", "2M outer plate", "9"]])
        self.assertEqual([r.requirement for r in boq_requirements(t)],
                         ["16A switch", "Plates (Coral Magna, Salmon Red) - 2M outer plate"])

    def test_single_numeric_location_column_is_not_a_quantity(self) -> None:
        t = table(["Block", "Floors", "Delivery within"], [["1", "3-4", "10 days"], ["2", "5-7", "25 days"]])
        self.assertEqual(t.kind, "other")

    def test_location_breakdown_and_mismatch_note(self) -> None:
        t = table(["Item", "Campus A", "Campus B", "Total"],
                  [["Plate", "5", "7", "12"], ["Box", "5", "5", "20"], ["Socket", "3", "-", ""]])
        a, b, c = boq_requirements(t)
        self.assertEqual((a.quantity, a.quantity_breakdown, a.note), (12, {"Campus A": 5, "Campus B": 7}, ""))
        self.assertEqual(b.quantity, 20)
        self.assertIn("add up to 10", b.note)
        self.assertEqual(c.quantity, 3)        # no total printed: the parts are summed

    def test_unit_written_in_the_quantity_cell(self) -> None:
        r = boq_requirements(table(["Particulars", "Quantity"], [["Extension board", "40 Sets"]]))[0]
        self.assertEqual((r.quantity, r.unit), (40, "Sets"))


class RenderTests(unittest.TestCase):
    def test_markdown_and_row_record(self) -> None:
        t = table(["Item", "Qty"], [["Switch | 6A", "10"]])
        self.assertEqual(table_markdown(t), "| Item | Qty |\n|---|---|\n| Switch / 6A | 10 |")
        self.assertEqual(row_record(t, t.rows[0], "Switches"),
                         "Table T1 (2. Scope, p.2) [Switches]: Item: Switch | 6A; Qty: 10")

    def test_key_value_table_renders_as_lines(self) -> None:
        t = table(["Tender No", "ABC/1"], [["EMD", "Rs 10,000"]])
        self.assertEqual(table_markdown(t), "Tender No: ABC/1\nEMD: Rs 10,000")


if __name__ == "__main__":
    unittest.main()
