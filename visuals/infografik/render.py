#!/usr/bin/env python3
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

PAGES = [
    "nachgestellt.html",
    "highlight-clavius.html",
    "highlight-palus-somni.html",
    "highlight-tycho-nordostteil.html",
]

OVERFLOW_JS = """
() => {
  const W = 1080, H = 1350;
  const hits = [];
  const all = document.querySelectorAll('body *');
  for (const el of all) {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    if (r.width === 0 && r.height === 0) continue;
    const text = (el.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 90);
    const clipped = el.scrollWidth > el.clientWidth + 2 || el.scrollHeight > el.clientHeight + 2;
    const outside = r.right > W + 1 || r.bottom > H + 1 || r.left < -1 || r.top < -1;
    if (clipped || outside) {
      hits.push({
        tag: el.tagName, cls: el.className, clipped, outside,
        box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)],
        scroll: [el.scrollWidth, el.clientWidth, el.scrollHeight, el.clientHeight],
        font: cs.fontSize, text
      });
    }
  }
  // also report the smallest font used on visible text nodes
  let minFont = 99, minEl = '';
  for (const el of all) {
    const t = (el.childNodes && [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim()));
    if (!t) continue;
    const s = parseFloat(getComputedStyle(el).fontSize);
    if (s && s < minFont) { minFont = s; minEl = (el.className||el.tagName) + ' ' + (el.innerText||'').slice(0,40); }
  }
  const pageH = document.querySelector('.page').getBoundingClientRect().height;
  const footer = document.querySelector('.footer');
  const main = document.querySelector('.main');
  let overlapFooter = false;
  let lastMain = null;
  if (main && footer) {
    const kids = [...main.children];
    lastMain = kids[kids.length-1].getBoundingClientRect();
    const fr = footer.getBoundingClientRect();
    overlapFooter = lastMain.bottom > fr.top + 1;
  }
  return { hits, minFont, minEl, pageH, lastMainBottom: lastMain && lastMain.bottom, footerTop: footer && footer.getBoundingClientRect().top, overlapFooter };
}
"""

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(
            viewport={"width": 1080, "height": 1350},
            device_scale_factor=2,
        )
        for name in PAGES:
            url = (ROOT / name).as_uri()
            page.goto(url, wait_until="networkidle")
            page.evaluate("() => document.fonts.ready")
            # give a tick for layout
            page.wait_for_timeout(150)
            info = page.evaluate(OVERFLOW_JS)
            out = OUT / (Path(name).stem + ".png")
            page.screenshot(path=str(out), type="png", full_page=False, omit_background=False)
            from PIL import Image
            im = Image.open(out)
            print(f"=== {name} -> {out.name} {im.size} ===")
            print(f"  minFont={info['minFont']}px  ({info['minEl']!r})")
            print(f"  pageH={info['pageH']} lastMainBottom={info['lastMainBottom']} footerTop={info['footerTop']} overlapFooter={info['overlapFooter']}")
            if info["hits"]:
                print(f"  OVERFLOW/OUTSIDE: {len(info['hits'])}")
                for h in info["hits"][:12]:
                    print("   ", h)
            else:
                print("  no overflow/outside")
        browser.close()

if __name__ == "__main__":
    main()
