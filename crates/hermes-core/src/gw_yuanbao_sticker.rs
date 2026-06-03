//! Yuanbao sticker (TIMFaceElem) support.
//!
//! Ported from `gateway/platforms/yuanbao_sticker.py`
//! (originally `yuanbao-openclaw-plugin/src/sticker/`).
//!
//! TIMFaceElem wire format:
//! ```json
//! {
//!     "msg_type": "TIMFaceElem",
//!     "msg_content": {
//!         "index": 0,
//!         "data": "<json>"
//!     }
//! }
//! ```
//!
//! The `data` field carries a JSON string with the sticker's metadata so the
//! receiver can look up the correct asset in the emoji pack.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::{json, Value};
use unicode_normalization::UnicodeNormalization;

/// A single built-in sticker entry (mirrors a row of `STICKER_MAP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sticker {
    pub sticker_id: &'static str,
    pub package_id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    pub width: u32,
    pub height: u32,
    pub formats: &'static str,
}

macro_rules! sticker {
    ($id:expr, $pkg:expr, $name:expr, $desc:expr) => {
        Sticker {
            sticker_id: $id,
            package_id: $pkg,
            name: $name,
            description: $desc,
            width: 128,
            height: 128,
            formats: "png",
        }
    };
}

/// Built-in sticker catalogue, in insertion order (matches the Python dict).
///
/// Note: `STICKER_MAP` in Python keys by canonical Chinese name; the Rust port
/// keeps an ordered slice and exposes name-based lookup helpers. Iteration
/// order is preserved to mirror Python dict insertion-order semantics, which
/// the fuzzy-search tie-breaking and `get_random_sticker` empty-query paths
/// rely on.
pub static STICKER_LIST: &[Sticker] = &[
    sticker!("278", "1003", "六六六", "666 厉害 牛 棒 绝了 好强 awesome"),
    sticker!("262", "1003", "我想开了", "想开 佛系 释怀 顿悟 看淡了 无所谓"),
    sticker!("130", "1003", "害羞", "腼腆 不好意思 脸红 娇羞 羞涩 捂脸"),
    sticker!("252", "1003", "比心", "笔芯 爱你 爱心手势 love heart 喜欢你"),
    sticker!("125", "1003", "委屈", "难过 想哭 可怜巴巴 瘪嘴 受伤 被欺负"),
    sticker!("146", "1003", "亲亲", "么么 mua 亲一下 kiss 飞吻 啵"),
    sticker!("131", "1003", "酷", "帅 墨镜 cool 高冷 有型 swagger"),
    sticker!("145", "1003", "睡", "睡觉 困 zzZ 打盹 躺平 休眠 sleepy"),
    sticker!("152", "1003", "发呆", "懵 愣住 放空 呆滞 出神 脑子空白"),
    sticker!("157", "1003", "可怜", "卖萌 求饶 委屈巴巴 弱小 拜托 眼巴巴"),
    sticker!("200", "1003", "摊手", "无奈 没办法 耸肩 随便 那咋整 whatever"),
    sticker!("213", "1003", "头大", "头疼 烦恼 郁闷 难搞 崩溃 一团乱"),
    sticker!("256", "1003", "吓", "害怕 惊恐 震惊 吓一跳 恐怖 怂"),
    sticker!("203", "1003", "吐血", "无语 崩溃 被雷 内伤 一口老血 屮"),
    sticker!("185", "1003", "哼", "傲娇 生气 不满 撇嘴 不理 赌气"),
    sticker!("220", "1003", "嘿嘿", "坏笑 猥琐笑 偷笑 憨笑 得意 你懂的"),
    sticker!("218", "1003", "头秃", "程序员 加班 焦虑 没头发 秃了 肝爆"),
    sticker!("221", "1003", "暗中观察", "窥屏 潜水 偷偷看 角落 围观 屏住呼吸"),
    sticker!("224", "1003", "我酸了", "嫉妒 柠檬精 羡慕 吃柠檬 眼红 恰柠檬"),
    sticker!("246", "1003", "打call", "应援 加油 支持 喝彩 助威 call"),
    sticker!("251", "1003", "庆祝", "祝贺 开心 耶 party 胜利 干杯"),
    sticker!("151", "1003", "奋斗", "努力 加油 拼搏 冲 干劲 卷起来"),
    sticker!("143", "1003", "惊讶", "震惊 哇 不敢相信 OMG 居然 这么离谱"),
    sticker!("144", "1003", "疑问", "问号 不懂 啥 为什么 啥情况 懵逼问"),
    sticker!("248", "1003", "仔细分析", "思考 推敲 认真 研究 琢磨 让我想想"),
    sticker!("184", "1003", "撅嘴", "嘟嘴 卖萌 不高兴 撒娇 嘴翘"),
    sticker!("199", "1003", "泪奔", "大哭 伤心 破防 感动哭 泪流满面 呜呜"),
    sticker!("276", "1003", "尊嘟假嘟", "真的假的 真假 可爱问 你骗我 是不是"),
    sticker!("113", "1003", "略略略", "调皮 吐舌 不服 略 气死你 鬼脸"),
    sticker!("180", "1003", "困", "想睡 倦 打哈欠 睁不开眼 好困啊 sleepy"),
    sticker!("181", "1003", "折磨", "难受 痛苦 煎熬 蚌埠住了 受不了 要命"),
    sticker!("182", "1003", "抠鼻", "不屑 无聊 淡定 无所谓 鄙视 挖鼻"),
    sticker!("183", "1003", "鼓掌", "拍手 叫好 赞同 666 喝彩 掌声"),
    sticker!("204", "1003", "斜眼笑", "滑稽 坏笑 doge 意味深长 阴阳怪气 嘿嘿嘿"),
    sticker!("216", "1003", "辣眼睛", "看不下去 cringe 毁三观 太丑了 瞎了"),
    sticker!("217", "1003", "哦哟", "惊讶 起哄 哇哦 有戏 不简单 哟"),
    sticker!("222", "1003", "吃瓜", "围观 看戏 八卦 路人 看热闹 板凳"),
    sticker!("225", "1003", "狗头", "doge 保命 开玩笑 滑稽 反讽 懂的都懂"),
    sticker!("227", "1003", "敬礼", "salute 尊重 收到 遵命 致敬 报告"),
    sticker!("231", "1003", "哦", "知道了 明白 敷衍 嗯 这样啊 收到"),
    sticker!("236", "1003", "拿到红包", "红包 谢谢老板 发财 开心 抢到了 欧气"),
    sticker!("239", "1003", "牛吖", "牛 厉害 强 666 佩服 大佬"),
    sticker!("272", "1003", "贴贴", "抱抱 亲昵 蹭蹭 亲密 靠靠 撒娇贴"),
    sticker!("138", "1003", "爱心", "心 love 喜欢你 红心 示爱 么么哒"),
    sticker!("170", "1003", "晚安", "好梦 睡了 night 早点休息 安啦 moon"),
    sticker!("176", "1003", "太阳", "晴天 早上好 阳光 morning 好天气 日"),
    sticker!("266", "1003", "柠檬", "酸 嫉妒 柠檬精 羡慕 我酸 恰柠檬"),
    sticker!("267", "1003", "大冤种", "倒霉 吃亏 自嘲 好心没好报 背锅 工具人"),
    sticker!("132", "1003", "吐了", "恶心 yue 受不了 嫌弃 想吐 生理不适"),
    sticker!("134", "1003", "怒", "生气 愤怒 火大 暴躁 气炸 怼"),
    sticker!("165", "1003", "玫瑰", "花 示爱 表白 浪漫 送你花 情人节"),
    sticker!("119", "1003", "凋谢", "花谢 失恋 难过 枯萎 心碎 凉了"),
    sticker!("159", "1003", "点赞", "赞 认同 好棒 good like 大拇指 顶"),
    sticker!("164", "1003", "握手", "合作 你好 商务 hello deal 成交 友好"),
    sticker!("163", "1003", "抱拳", "谢谢 失敬 江湖 承让 拜托 有礼"),
    sticker!("169", "1003", "ok", "好的 收到 没问题 okay 行 可以 懂了"),
    sticker!("174", "1003", "拳头", "加油 干 冲 fight 力量 击拳 硬气"),
    sticker!("191", "1003", "鞭炮", "过年 喜庆 爆竹 春节 噼里啪啦 红"),
    sticker!("258", "1003", "烟花", "庆典 漂亮 新年 嘭 绽放 节日快乐"),
];

/// Name-keyed view over `STICKER_LIST` (mirrors the Python `STICKER_MAP` dict).
/// Returns the first entry for a given name (names are unique in the catalogue).
pub fn sticker_map() -> &'static HashMap<&'static str, &'static Sticker> {
    static MAP: OnceLock<HashMap<&'static str, &'static Sticker>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::with_capacity(STICKER_LIST.len());
        for s in STICKER_LIST {
            m.entry(s.name).or_insert(s);
        }
        m
    })
}

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

/// Find a sticker by name with fuzzy fallback.
///
/// Match priority:
///   1. exact name equality
///   2. name contains query, or query contains name (substring either way)
///   3. description contains query (synonym search)
///   4. generic fuzzy score (same algorithm as `search_stickers`); the
///      top-scoring entry is returned if any.
///
/// Returns `None` when `name` is empty/whitespace or nothing matches.
pub fn get_sticker_by_name(name: &str) -> Option<&'static Sticker> {
    if name.is_empty() {
        return None;
    }
    let query = name.trim();
    if query.is_empty() {
        return None;
    }

    // 1. exact
    if let Some(s) = sticker_map().get(query) {
        return Some(*s);
    }

    // 2. substring either direction (iterate in catalogue order)
    for s in STICKER_LIST {
        if s.name.contains(query) || query.contains(s.name) {
            return Some(s);
        }
    }

    // 3. description contains query
    for s in STICKER_LIST {
        if s.description.contains(query) {
            return Some(s);
        }
    }

    // 4. fuzzy
    search_stickers(query, 1).into_iter().next()
}

/// Find a sticker by exact `sticker_id`. Returns `None` for empty input or no match.
pub fn get_sticker_by_id(sticker_id: &str) -> Option<&'static Sticker> {
    if sticker_id.is_empty() {
        return None;
    }
    let sid = sticker_id.trim();
    if sid.is_empty() {
        return None;
    }
    STICKER_LIST.iter().find(|s| s.sticker_id == sid)
}

/// Pick a random sticker, optionally restricted to a category keyword.
///
/// When `category` is `Some`, candidates are those whose `description` or
/// `name` contains the keyword; if none match, falls back to the full table.
/// `category == None` picks from the full table.
///
/// `rand_index` must yield a value in `[0, len)` given the candidate count;
/// callers supply their own RNG (the workspace does not depend on `rand`).
pub fn get_random_sticker_with<F>(category: Option<&str>, mut rand_index: F) -> &'static Sticker
where
    F: FnMut(usize) -> usize,
{
    if let Some(cat) = category {
        if !cat.is_empty() {
            let candidates: Vec<&'static Sticker> = STICKER_LIST
                .iter()
                .filter(|s| s.description.contains(cat) || s.name.contains(cat))
                .collect();
            if !candidates.is_empty() {
                let i = rand_index(candidates.len());
                return candidates[i];
            }
        }
    }
    let i = rand_index(STICKER_LIST.len());
    &STICKER_LIST[i]
}

/// Convenience wrapper using a simple time-seeded selection.
///
/// Mirrors `get_random_sticker(category=None)` / `random.choice`. Uses a
/// nanosecond clock as the entropy source to avoid pulling in `rand`.
pub fn get_random_sticker(category: Option<&str>) -> &'static Sticker {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    get_random_sticker_with(category, |len| (seed as usize).wrapping_mul(2654435761) % len)
}

// ---------------------------------------------------------------------------
// Fuzzy search (aligned with sticker-cache.ts `searchStickers`)
// ---------------------------------------------------------------------------

/// Punctuation/whitespace removed by `_compact_text`.
/// Matches the Python regex `[\s　\-_·.,，。!！?？"“”'‘’、/\\]+`.
fn is_punct_char(c: char) -> bool {
    if c.is_whitespace() {
        return true;
    }
    matches!(
        c,
        '\u{3000}'   // ideographic space
            | '-' | '_' | '·' | '.' | ',' | '，' | '。'
            | '!' | '！' | '?' | '？'
            | '"' | '“' | '”'
            | '\'' | '‘' | '’'
            | '、' | '/' | '\\'
    )
}

/// `unicodedata.normalize("NFKC", raw).strip().lower()`.
pub fn normalize_text(raw: &str) -> String {
    raw.nfkc().collect::<String>().trim().to_lowercase()
}

/// Strip punctuation/whitespace from the normalized form.
pub fn compact_text(raw: &str) -> String {
    normalize_text(raw)
        .chars()
        .filter(|c| !is_punct_char(*c))
        .collect()
}

/// Fraction of `needle` chars covered by `haystack` as a multiset.
fn multiset_char_hit_ratio(needle: &str, haystack: &str) -> f64 {
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.is_empty() {
        return 0.0;
    }
    let mut bag: HashMap<char, i64> = HashMap::new();
    for ch in haystack.chars() {
        *bag.entry(ch).or_insert(0) += 1;
    }
    let mut hits = 0usize;
    for ch in &needle_chars {
        let n = bag.get(ch).copied().unwrap_or(0);
        if n > 0 {
            hits += 1;
            bag.insert(*ch, n - 1);
        }
    }
    hits as f64 / needle_chars.len() as f64
}

/// Bigram Jaccard similarity over character bigrams.
fn bigram_jaccard(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    if a_chars.len() < 2 || b_chars.len() < 2 {
        return 0.0;
    }
    let bigrams = |chars: &[char]| -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::new();
        for w in chars.windows(2) {
            set.insert(w.iter().collect::<String>());
        }
        set
    };
    let set_a = bigrams(&a_chars);
    let set_b = bigrams(&b_chars);
    let inter = set_a.intersection(&set_b).count();
    let union = set_a.len() + set_b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// Ratio of `needle` matched as an in-order subsequence of `haystack`.
fn longest_subsequence_ratio(needle: &str, haystack: &str) -> f64 {
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.is_empty() {
        return 0.0;
    }
    let mut j = 0usize;
    for ch in haystack.chars() {
        if j >= needle_chars.len() {
            break;
        }
        if ch == needle_chars[j] {
            j += 1;
        }
    }
    j as f64 / needle_chars.len() as f64
}

/// Score a single field against a query (mirrors `_score_field`).
fn score_field(haystack: &str, query: &str) -> f64 {
    let hay = normalize_text(haystack);
    let q = normalize_text(query);
    if hay.is_empty() || q.is_empty() {
        return 0.0;
    }
    let hay_c = compact_text(haystack);
    let q_c = compact_text(query);
    let mut best = 0.0_f64;

    if hay == q {
        best = best.max(100.0);
    }
    if hay.contains(&q) {
        let q_len = q.chars().count() as f64;
        best = best.max(92.0 + q_len.min(6.0));
    }
    if q.chars().count() >= 2 && hay.starts_with(&q) {
        best = best.max(88.0);
    }
    if !q_c.is_empty() && hay_c.contains(&q_c) {
        best = best.max(86.0);
    }
    best = best.max(multiset_char_hit_ratio(&q_c, &hay_c) * 62.0);
    best = best.max(bigram_jaccard(&q_c, &hay_c) * 58.0);
    best = best.max(longest_subsequence_ratio(&q_c, &hay_c) * 52.0);
    if q.chars().count() == 1 && hay.contains(&q) {
        best = best.max(68.0);
    }
    best
}

/// Fuzzy-search the built-in catalogue, returning up to `limit` results sorted
/// by descending score (mirrors `search_stickers`).
///
/// Empty/whitespace queries return the first `limit` entries in catalogue order.
pub fn search_stickers(query: &str, limit: usize) -> Vec<&'static Sticker> {
    let safe_limit = limit_clamp(limit);

    if query.is_empty() || normalize_text(query).is_empty() {
        return STICKER_LIST.iter().take(safe_limit).collect();
    }

    let q_norm = normalize_text(query);

    let mut scored: Vec<(f64, &'static Sticker)> = Vec::with_capacity(STICKER_LIST.len());
    for s in STICKER_LIST {
        let name_s = score_field(s.name, query);
        let desc_s = score_field(s.description, query) * 0.88;
        let sid = s.sticker_id.trim();
        let mut id_s = 0.0_f64;
        if !sid.is_empty() && !q_norm.is_empty() {
            let sid_norm = normalize_text(sid);
            if sid_norm == q_norm {
                id_s = 100.0;
            } else if sid_norm.contains(&q_norm) {
                id_s = 84.0;
            }
        }
        let best = name_s.max(desc_s).max(id_s);
        scored.push((best, s));
    }

    // Stable descending sort by score (preserves catalogue order on ties,
    // matching Python's stable `list.sort`).
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top = scored.first().map(|p| p.0).unwrap_or(0.0);
    if top <= 0.0 {
        return scored.into_iter().take(safe_limit).map(|(_, s)| s).collect();
    }

    let floor = if top >= 22.0 {
        18.0
    } else if top >= 12.0 {
        (top * 0.5).max(10.0)
    } else {
        (top * 0.35).max(6.0)
    };

    let filtered: Vec<(f64, &'static Sticker)> =
        scored.iter().filter(|p| p.0 >= floor).copied().collect();
    let out = if filtered.is_empty() { scored } else { filtered };
    out.into_iter().take(safe_limit).map(|(_, s)| s).collect()
}

/// `max(1, min(500, limit or 10))` — clamp matching the Python default.
fn limit_clamp(limit: usize) -> usize {
    let l = if limit == 0 { 10 } else { limit };
    l.clamp(1, 500)
}

// ---------------------------------------------------------------------------
// Wire-format builders
// ---------------------------------------------------------------------------

/// Build a TIMFaceElem `msg_body` list.
///
/// Yuanbao convention:
///   - `index` is normally `0` (the server identifies the emoji via `data`);
///     a non-zero `face_index` is treated as a legacy QQ-emoji id and passed
///     through directly.
///   - `data` is a JSON string carrying `sticker_id` / `package_id` etc.;
///     when `None`, only `index` is emitted.
///
/// Returns a JSON array, e.g.
/// `[{"msg_type":"TIMFaceElem","msg_content":{"index":0,"data":"..."}}]`.
pub fn build_face_msg_body(face_index: i64, data: Option<&str>) -> Value {
    let mut msg_content = serde_json::Map::new();
    msg_content.insert("index".to_string(), json!(face_index));
    if let Some(d) = data {
        msg_content.insert("data".to_string(), json!(d));
    }
    json!([{
        "msg_type": "TIMFaceElem",
        "msg_content": Value::Object(msg_content),
    }])
}

/// Build a TIMFaceElem `msg_body` directly from a catalogue `Sticker`.
///
/// The serialized `data` payload uses compact separators (`,`/`:`) and
/// preserves field order to stay byte-compatible with the original JS plugin.
pub fn build_sticker_msg_body(sticker: &Sticker) -> Value {
    let data_payload = serialize_sticker_data(sticker);
    build_face_msg_body(0, Some(&data_payload))
}

/// Serialize a sticker's `data` payload as compact JSON (no spaces), in the
/// exact key order used by the Python `json.dumps(..., separators=(",", ":"))`
/// call: sticker_id, package_id, width, height, formats, name.
pub fn serialize_sticker_data(sticker: &Sticker) -> String {
    // Build manually to guarantee key ordering and compact separators.
    let mut out = String::with_capacity(96);
    out.push('{');
    out.push_str("\"sticker_id\":");
    push_json_string(&mut out, sticker.sticker_id);
    out.push_str(",\"package_id\":");
    push_json_string(&mut out, sticker.package_id);
    out.push_str(",\"width\":");
    out.push_str(&sticker.width.to_string());
    out.push_str(",\"height\":");
    out.push_str(&sticker.height.to_string());
    out.push_str(",\"formats\":");
    push_json_string(&mut out, sticker.formats);
    out.push_str(",\"name\":");
    push_json_string(&mut out, sticker.name);
    out.push('}');
    out
}

/// Append a JSON-escaped string literal (with surrounding quotes), leaving
/// non-ASCII characters as-is to mirror `ensure_ascii=False`.
fn push_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_has_all_entries_and_unique_names() {
        assert_eq!(STICKER_LIST.len(), 59);
        assert_eq!(sticker_map().len(), STICKER_LIST.len());
    }

    #[test]
    fn exact_name_lookup() {
        let s = get_sticker_by_name("六六六").unwrap();
        assert_eq!(s.sticker_id, "278");
    }

    #[test]
    fn empty_name_returns_none() {
        assert!(get_sticker_by_name("").is_none());
        assert!(get_sticker_by_name("   ").is_none());
    }

    #[test]
    fn substring_either_direction() {
        // query contains a sticker name ("酷" is the "酷" sticker)
        let s = get_sticker_by_name("好酷啊").unwrap();
        assert_eq!(s.name, "酷");
    }

    #[test]
    fn description_match() {
        // "awesome" only appears in 六六六's description
        let s = get_sticker_by_name("awesome").unwrap();
        assert_eq!(s.sticker_id, "278");
    }

    #[test]
    fn lookup_by_id() {
        assert_eq!(get_sticker_by_id("252").unwrap().name, "比心");
        assert!(get_sticker_by_id("").is_none());
        assert!(get_sticker_by_id("99999").is_none());
        assert_eq!(get_sticker_by_id("  252  ").unwrap().name, "比心");
    }

    #[test]
    fn search_empty_returns_prefix() {
        let r = search_stickers("", 3);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].name, "六六六");
        assert_eq!(r[1].name, "我想开了");
    }

    #[test]
    fn search_by_id_ranks_top() {
        let r = search_stickers("252", 3);
        assert!(!r.is_empty());
        assert_eq!(r[0].sticker_id, "252");
    }

    #[test]
    fn search_compacts_spaced_query() {
        let r = search_stickers("暗 中 观 察", 3);
        assert!(!r.is_empty());
        assert_eq!(r[0].name, "暗中观察");
    }

    #[test]
    fn fullwidth_query_normalized() {
        // Fullwidth "ＯＫ" -> NFKC "OK" -> lower "ok"
        let r = search_stickers("ＯＫ", 3);
        assert!(!r.is_empty());
        assert_eq!(r[0].name, "ok");
    }

    #[test]
    fn limit_clamping() {
        assert_eq!(limit_clamp(0), 10);
        assert_eq!(limit_clamp(5), 5);
        assert_eq!(limit_clamp(9999), 500);
    }

    #[test]
    fn normalize_and_compact() {
        assert_eq!(normalize_text("  Hello  "), "hello");
        assert_eq!(compact_text("a b-c_d.e"), "abcde");
    }

    #[test]
    fn face_body_without_data() {
        let body = build_face_msg_body(5, None);
        let arr = body.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["msg_type"], "TIMFaceElem");
        assert_eq!(arr[0]["msg_content"]["index"], 5);
        assert!(arr[0]["msg_content"].get("data").is_none());
    }

    #[test]
    fn face_body_with_data() {
        let body = build_face_msg_body(0, Some("payload"));
        let arr = body.as_array().unwrap();
        assert_eq!(arr[0]["msg_content"]["index"], 0);
        assert_eq!(arr[0]["msg_content"]["data"], "payload");
    }

    #[test]
    fn sticker_data_payload_is_compact_and_ordered() {
        let s = get_sticker_by_id("278").unwrap();
        let payload = serialize_sticker_data(s);
        assert_eq!(
            payload,
            r#"{"sticker_id":"278","package_id":"1003","width":128,"height":128,"formats":"png","name":"六六六"}"#
        );
        // round-trips as valid JSON
        let v: Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(v["sticker_id"], "278");
        assert_eq!(v["width"], 128);
    }

    #[test]
    fn sticker_msg_body_wraps_payload() {
        let s = get_sticker_by_id("278").unwrap();
        let body = build_sticker_msg_body(s);
        let arr = body.as_array().unwrap();
        assert_eq!(arr[0]["msg_content"]["index"], 0);
        let data = arr[0]["msg_content"]["data"].as_str().unwrap();
        assert!(data.contains("\"sticker_id\":\"278\""));
    }

    #[test]
    fn random_with_category_filters() {
        // "sleepy" appears in 睡 and 困 descriptions
        let s = get_random_sticker_with(Some("sleepy"), |_| 0);
        assert!(s.description.contains("sleepy"));
    }

    #[test]
    fn random_no_category_picks_some() {
        let s = get_random_sticker_with(None, |_| 0);
        assert_eq!(s.name, "六六六");
    }

    #[test]
    fn scoring_exact_field() {
        assert_eq!(score_field("六六六", "六六六"), 100.0);
    }

    #[test]
    fn fuzzy_no_match_still_returns_results() {
        // gibberish: top score <= 0 path should still return up to limit
        let r = search_stickers("zzzzzzqxqx", 2);
        assert!(r.len() <= 2);
    }
}
