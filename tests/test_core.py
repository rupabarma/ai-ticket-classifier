"""
Core logic tests (no web server needed). Run from the project root:

    python -m unittest tests.test_core -v
"""
import unittest

from app.model import REVIEW_LABEL, TicketClassifier
from app.preprocessing import clean_text, preprocess_ticket
from app.priority import detect_priority


class PreprocessingTests(unittest.TestCase):
    def test_lowercase_urls_symbols_whitespace(self):
        text = "  URGENT!!  See https://example.com/x?y=1 and www.test.org   NOW #$% "
        self.assertEqual(clean_text(text), "urgent see and now")

    def test_none_and_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(preprocess_ticket("", ""), "")

    def test_subject_and_body_are_combined(self):
        self.assertEqual(preprocess_ticket("Hello", "World"), "hello world")


class PriorityTests(unittest.TestCase):
    def test_each_default_keyword(self):
        for phrase in ["urgent", "the site is down", "login not working", "critical bug", "fix immediately"]:
            self.assertEqual(detect_priority("", phrase).label, "Urgent", phrase)

    def test_normal(self):
        self.assertEqual(detect_priority("Question", "How do I change my invoice address?").label, "Normal")

    def test_whole_word_only(self):
        self.assertEqual(detect_priority("Download guide", "Where can I download it?").label, "Normal")

    def test_case_insensitive_and_subject_checked(self):
        self.assertEqual(detect_priority("URGENT", "hello").label, "Urgent")

    def test_extendable(self):
        self.assertEqual(detect_priority("", "system outage", keywords=["outage"]).label, "Urgent")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = TicketClassifier()

    def test_department_examples(self):
        cases = {
            "Billing": ("Charged twice", "I was billed twice for my subscription, please refund the duplicate payment."),
            "Technical": ("App crash", "The app crashes when I open it and the login page shows an error."),
            "HR": ("Leave request", "I would like to apply for annual leave and check my payslip and salary."),
            "General": ("Request a demo", "We would like a demo and information about your pricing for our team."),
        }
        for expected, (subject, body) in cases.items():
            self.assertEqual(self.clf.predict(subject, body).department, expected)

    def test_probabilities_sum_to_one(self):
        p = self.clf.predict("Refund", "Please refund my last payment")
        self.assertAlmostEqual(sum(p.probabilities.values()), 1.0, places=6)

    def test_low_confidence_needs_review(self):
        p = self.clf.predict("xqzv", "zzzz qqqq wwww")
        self.assertTrue(p.needs_review)
        self.assertEqual(p.display_department, REVIEW_LABEL)

    def test_high_confidence_is_assigned(self):
        p = self.clf.predict("Charged twice", "I was billed twice and want a refund of the duplicate charge.")
        self.assertFalse(p.needs_review)
        self.assertGreaterEqual(p.confidence, 0.60)

    def test_no_readable_text_raises(self):
        with self.assertRaises(ValueError):
            self.clf.predict("!!!", "???")


if __name__ == "__main__":
    unittest.main()
