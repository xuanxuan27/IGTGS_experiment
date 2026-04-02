import re
import json
import time
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# --- Regex 配置 ---
RE_XN = re.compile(r"x\s*(\d+)", re.IGNORECASE)
RE_REPEAT_BLOCK = re.compile(r"\|\|\s*(.*?)\s*\|\|\s*x\s*(\d+)", re.IGNORECASE | re.DOTALL)
RE_RETURN = re.compile(r"[（(]\s*回\s*([▲★☆\d])\s*[)）]")
RE_JUMP_TO = re.compile(r"[（(]\s*接\s*([▲★☆\d])\s*[)）]")
RE_SECTION = re.compile(r"[（(]\s*(\d+)\s*[)）]")
RE_VERSE_NUM = re.compile(r"(\d+)\s*\.")
RE_INTERLUDE = re.compile(r"間奏\s*([0-9一二三四五六七八九十]+)")
RE_CHORD_TOKEN = re.compile(
    r"^[A-G][#b]?(?:[a-zA-Z0-9+#\-]*)(?:/[A-G][#b]?)?(?:[－-])?$"
)


def parse_cn_number(token):
    """解析簡單中文數字（支援 一~十九）"""
    if token.isdigit():
        return int(token)
    if token == "十":
        return 10
    cn_map = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9
    }
    if "十" in token:
        left, _, right = token.partition("十")
        tens = cn_map.get(left, 1) if left else 1
        ones = cn_map.get(right, 0) if right else 0
        return tens * 10 + ones
    return cn_map.get(token)


def get_interlude_no(text):
    m = RE_INTERLUDE.search(text)
    if not m:
        return None
    return parse_cn_number(m.group(1))


def parse_chords_from_text(text):
    text = text.replace("｜", "|")
    tokens = re.split(r"[\s|]+", text)
    chords = []
    for token in tokens:
        t = token.strip().strip("()（）[]【】,，。:：;；")
        if not t:
            continue
        if RE_CHORD_TOKEN.match(t):
            chords.append(t)
    return chords


def expand_repeat_blocks_in_text(text):
    """將 || ... || xN 區塊展開為實際重複和弦字串。"""
    def repl(match):
        inner = match.group(1)
        count = int(match.group(2))
        inner_chords = parse_chords_from_text(inner)
        if not inner_chords:
            return " "
        return " " + " ".join(inner_chords * count) + " "

    return RE_REPEAT_BLOCK.sub(repl, text)


def detect_line_repeat_x(text):
    """偵測整行 xN（排除 ||...|| xN 的區塊重複語意）。"""
    text_wo_repeat_block = RE_REPEAT_BLOCK.sub(" ", text)
    m = RE_XN.search(text_wo_repeat_block)
    return int(m.group(1)) if m else 1

class GuitarTabLogicEngine:
    def __init__(self, lines, debug=False):
        self.lines = lines
        self.debug = debug
        self.final_chords = []
        self.current_verse = 1
        self.visited_returns = set()
        self.anchors = self._map_anchors()
        self.pending_skip_interlude_no = None
        self.pending_skip_to_section = None

    def _map_anchors(self):
        """修正問題 2：錨點應精確定位到和弦行"""
        mapping = {}

        def nearest_prev_music_idx(start_idx):
            """找出 start_idx 當前或上方最近的和弦行索引。"""
            j = start_idx
            while j >= 0:
                if self.lines[j]["type"] == "music":
                    return j
                j -= 1
            return start_idx

        def nearest_next_music_idx(start_idx):
            """找出 start_idx 下方最近的和弦行索引。"""
            j = start_idx + 1
            while j < len(self.lines):
                if self.lines[j]["type"] == "music":
                    return j
                j += 1
            return nearest_prev_music_idx(start_idx)

        for i, line in enumerate(self.lines):
            txt = line['text']
            for sym in ["▲", "★", "☆"]:
                if sym in txt and "回" not in txt and "接" not in txt:
                    # 無論中間隔幾行歌詞，都定位到最近的上方和弦行
                    mapping[sym] = nearest_prev_music_idx(i)
            
            m_sec = RE_SECTION.search(txt)
            if m_sec and "回" not in txt and "接" not in txt:
                sec_num = m_sec.group(1)
                # 段落標記 (1)(2)(3) 應接到標記下方第一行和弦
                mapping[sec_num] = nearest_next_music_idx(i)
        return mapping

    def run(self):
        curr = 0
        total_lines = len(self.lines)
        steps = 0
        current_interlude_no = None
        
        while curr < total_lines and steps < 2000:
            steps += 1
            line = self.lines[curr]
            txt = line["text"]
            state_key = (
                curr,
                self.current_verse,
                self.pending_skip_interlude_no,
                self.pending_skip_to_section,
            )
            same_state_hits = getattr(self, "_state_hits", {})
            same_state_hits[state_key] = same_state_hits.get(state_key, 0) + 1
            self._state_hits = same_state_hits
            if same_state_hits[state_key] > 6:
                if self.debug:
                    print(f"!! 偵測到重複狀態 {state_key}，強制前進避免迴圈")
                curr += 1
                continue

            # 追蹤目前是否在「間奏N」區塊
            interlude_no = get_interlude_no(txt)
            if interlude_no is not None:
                current_interlude_no = interlude_no
            m_sec = RE_SECTION.search(txt)
            if current_interlude_no is not None and m_sec:
                # 看到下一段段落標記，代表離開間奏區塊
                if int(m_sec.group(1)) >= current_interlude_no + 1:
                    current_interlude_no = None

            # 若目前段落已往後（例如 V3），遇到舊 section（例如 (1)/(2)）就直接前跳
            if m_sec:
                sec_num = int(m_sec.group(1))
                curr_sec_key = str(self.current_verse)
                if (
                    sec_num < self.current_verse
                    and curr_sec_key in self.anchors
                    and self.anchors[curr_sec_key] > curr
                ):
                    if self.debug:
                        print(f"-> 略過舊段落 ({sec_num})，直接接 ({self.current_verse})")
                    current_interlude_no = None
                    curr = self.anchors[curr_sec_key]
                    continue

            # 間奏N 的 (回▲) 視為進入下一段；下一次回到同一個間奏N，直接跳到 (N+1)
            if (
                self.pending_skip_interlude_no is not None
                and current_interlude_no == self.pending_skip_interlude_no
                and self.pending_skip_to_section in self.anchors
            ):
                target_idx = self.anchors[self.pending_skip_to_section]
                if target_idx > curr:
                    if self.debug:
                        print(f"-> 跳過間奏{self.pending_skip_interlude_no}，直接接 ({self.pending_skip_to_section})")
                    self.pending_skip_interlude_no = None
                    self.pending_skip_to_section = None
                    current_interlude_no = None
                    curr = target_idx
                    continue
            
            # --- 核心過濾邏輯 ---
            # 如果這行有明確指定段落，且當前段落不在其中，則跳過
            if line["belong_to_verses"] and self.current_verse not in line["belong_to_verses"]:
                curr += 1
                continue

            # 彈奏和弦
            if line["chords"]:
                if self.debug: print(f"L{curr} [V{self.current_verse}]: 彈奏 {line['chords']}")
                self.final_chords.extend(line["chords"] * line.get("repeat_x", 1))

            # 指令判斷
            m_jump = RE_JUMP_TO.search(txt)
            if m_jump:
                target = m_jump.group(1)
                if target in self.anchors:
                    if self.debug: print(f"-> 接 {target}")
                    # 接續跳轉後，舊的間奏上下文失效，避免誤判為仍在間奏區塊
                    current_interlude_no = None
                    curr = self.anchors[target]
                    continue

            m_ret = RE_RETURN.search(txt)
            if m_ret:
                target = m_ret.group(1)
                call_id = (curr, target)
                if target in self.anchors and call_id not in self.visited_returns:
                    self.visited_returns.add(call_id)
                    self.current_verse += 1  # 增加段落數
                    if target == "▲" and current_interlude_no is not None:
                        self.pending_skip_interlude_no = current_interlude_no
                        self.pending_skip_to_section = str(current_interlude_no + 1)
                    if self.debug: print(f"-> 回 {target}, 進入段落 {self.current_verse}")
                    # 回跳後離開原位置，間奏上下文要清空，避免下一輪立刻觸發跳過間奏
                    current_interlude_no = None
                    curr = self.anchors[target]
                    continue

            curr += 1
        return self.final_chords

def get_verse_ids(p):
    """從 <span> 提取所有段落編號"""
    txt = "".join([s.get_text(strip=True) for s in p.select("span")[:10]])
    return [int(n) for n in RE_VERSE_NUM.findall(txt)]

def create_chrome_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # 在 Linux 上優先使用系統可找到的瀏覽器，降低「Chrome instance exited」機率
    chrome_binary = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if chrome_binary:
        options.binary_location = chrome_binary

    return webdriver.Chrome(options=options)

def scrape_and_parse(url):
    driver = create_chrome_driver()

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "tone_z")))
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "lxml")
        p_tags = soup.select("#tone_z p")

        raw_data = []
        for p in p_tags:
            cls = p.get("class", [])
            txt = p.get_text(" ", strip=False)
            expanded_txt = expand_repeat_blocks_in_text(txt)
            chords_from_span = [
                re.sub(r"\s+", "", s.get_text("", strip=True))
                for s in p.select("span.tf")
                if s.get_text("", strip=True)
            ]
            chords = chords_from_span

            # 只有在含有 ||...|| xN 時，才用文字展開結果覆蓋，避免正常和弦被拆壞（如 E♭dim7）
            if RE_REPEAT_BLOCK.search(txt):
                chords_from_text = parse_chords_from_text(expanded_txt)
                if chords_from_text:
                    chords = chords_from_text
            elif not chords:
                chords = parse_chords_from_text(expanded_txt)
            
            raw_data.append({
                "type": "music" if "music" in cls else "lyric",
                "text": txt,
                "chords": chords,
                "verse_ids": get_verse_ids(p),
                "belong_to_verses": set(), # 最終判斷的歸屬
                "repeat_x": 1
            })
            raw_data[-1]["repeat_x"] = detect_line_repeat_x(txt)

        # --- 垂直關聯繫結 (核心修正) ---
        for i in range(len(raw_data)):
            # 如果是歌詞行，它本身就有 verse_ids
            if raw_data[i]["type"] == "lyric":
                raw_data[i]["belong_to_verses"].update(raw_data[i]["verse_ids"])
            
            # 如果是和弦行，往後找緊跟著的歌詞行
            if raw_data[i]["type"] == "music":
                j = i + 1
                while j < len(raw_data) and raw_data[j]["type"] == "lyric":
                    # 和弦行繼承下方所有歌詞行的段落編號
                    raw_data[i]["belong_to_verses"].update(raw_data[j]["verse_ids"])
                    j += 1
                
                # 如果這行和弦下方完全沒有編號歌詞，代表它是全域通用的 (如前奏、間奏)
                # 這裡保留為空集，引擎會判定為通用

        engine = GuitarTabLogicEngine(raw_data, debug=True)
        return engine.run()

    finally:
        driver.quit()

if __name__ == "__main__":
    url = input("91pu URL: ").strip()
    if url:
        result = scrape_and_parse(url)
        output_path = input("輸出 JSON 檔名（預設 scraped.json）: ").strip() or "scraped.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"source": url, "chords": result}, f, ensure_ascii=False, indent=2)
        print(f"\n解析成功，總和弦序列長度: {len(result)}，已儲存到 {output_path}")