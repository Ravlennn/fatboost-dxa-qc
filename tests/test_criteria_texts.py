"""The criterion cards must not show a TZ number next to a verdict that the number contradicts.

Example from the image review (20.10): a right hip with ROI margins 24/37/36/35 mm, the TZ line
«3 см сверху и снизу, 2 см сбоку» and a green ROI status. The release logic flags ROI only when the
ROI frame leaves the image (release_predictor.roi_edge_score); the cards must say so.
"""
import re
import unittest
from pathlib import Path

from dxaqc.release_predictor import roi_edge_score

ROOT = Path(__file__).resolve().parents[1]
UIS = ('ui', 'ui-journal')
CODES = ('spine_scan_range', 'spine_axis_tilt', 'spine_artifact', 'hip_positioning', 'hip_roi_margins')


def criteria(ui):
    text = (ROOT / ui / 'src' / 'labels.ts').read_text(encoding='utf-8')
    rows = re.findall(r"\{ code: '(\w+)', title: '([^']*)', rule: '([^']*)', decides: '([^']*)' \}", text)
    return {code: {'title': title, 'rule': rule, 'decides': decides} for code, title, rule, decides in rows}


class CriterionTexts(unittest.TestCase):
    def test_every_criterion_says_how_it_is_decided(self):
        for ui in UIS:
            found = criteria(ui)
            self.assertEqual(set(found), set(CODES), ui)
            for code, c in found.items():
                self.assertTrue(c['decides'].strip(), (ui, code))

    def test_roi_card_explains_the_frame_rule_that_the_model_applies(self):
        # 24 mm is below the TZ's 30 mm, yet it is not a defect: only a frame past the image edge is
        margins = {'roi_top_mm': 24.0, 'roi_bottom_mm': 37.0, 'roi_left_mm': 36.0, 'roi_right_mm': 35.0}
        self.assertLess(roi_edge_score(margins)[1], 0.5)
        self.assertGreater(roi_edge_score({**margins, 'roi_top_mm': -3.0})[1], 0.5)
        for ui in UIS:
            roi = criteria(ui)['hip_roi_margins']
            self.assertIn('3 см', roi['rule'])
            self.assertIn('край снимка', roi['decides'], ui)

    def test_cards_label_the_tz_line_and_show_the_decision_line(self):
        for ui in UIS:
            viewer = (ROOT / ui / 'src' / 'components' / 'Viewer.tsx').read_text(encoding='utf-8')
            self.assertIn('По ТЗ:', viewer, ui)
            self.assertIn('{c.decides}', viewer, ui)
            self.assertNotIn('Норма:</span>', viewer, ui)   # the TZ line is a reference, not the verdict

    def test_built_interfaces_carry_the_explanation(self):
        roi = criteria('ui-journal')['hip_roi_margins']['decides']
        for dist in ('webui_dist', 'webui_journal_dist'):
            bundles = list((ROOT / 'src' / 'dxaqc' / dist / 'assets').glob('*.js'))
            if not bundles:
                self.skipTest(f'{dist} is not built')
            self.assertTrue(any(roi in b.read_text(encoding='utf-8') for b in bundles), f'{dist}: rebuild the UI')


if __name__ == '__main__':
    unittest.main()
