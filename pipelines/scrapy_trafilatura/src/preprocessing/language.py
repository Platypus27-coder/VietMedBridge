from __future__ import annotations

import re

VI_ACCENTS = frozenset('ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ')


def detect_language(text: str, config: dict) -> dict:
    if config['language']['backend'] != 'heuristic':
        raise ValueError('Only benchmarked heuristic LID is enabled; configure/install a tested model adapter before changing backend.')
    sample = text[:config['language']['sample_chars']]
    letters = [c for c in sample if c.isalpha()]
    denominator = max(len(letters), 1)
    cjk = sum('\u4e00' <= c <= '\u9fff' for c in letters)/denominator
    vietnamese = sum(c in VI_ACCENTS for c in letters)/denominator
    japanese = any('\u3040' <= c <= '\u30ff' and c.isalpha() and c not in ('\u30fb', '\u30fc') for c in sample)
    korean = any('\uac00' <= c <= '\ud7af' for c in sample)
    words = set(re.findall(r'\w+', sample.lower()))
    vi_words = len(words & {'và', 'của', 'bệnh', 'điều', 'trị', 'với', 'người', 'không'})
    en_words = len(words & {'the', 'and', 'with', 'of', 'for', 'patient', 'treatment', 'disease'})
    if len(letters) < 20:
        label = 'unknown'
    elif japanese or korean:
        label = 'other'
    elif cjk > 0.15 and (vietnamese > 0.03 or en_words >= 3):
        label = 'mixed'
    elif cjk > 0.4:
        label = 'zh'
    elif vietnamese > 0.02 and vi_words >= 2:
        label = 'mixed' if en_words >= 5 and vi_words >= 4 else 'vi'
    elif en_words >= 3 and cjk < 0.05:
        label = 'en'
    else:
        label = config['language']['low_confidence_label']
    return {'language': label, 'language_confidence': None, 'language_method': 'heuristic-v2'}
