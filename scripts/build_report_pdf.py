"""Build KRMU-formatted academic Capstone PDF report from HTML using Playwright (Edge/Chrome) or Edge CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Render an HTML document to an A4 academic PDF using CSS @page margin rules."""
    html_uri = html_path.resolve().as_uri()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome", None):
                try:
                    kwargs = {"headless": True}
                    if channel is not None:
                        kwargs["channel"] = channel
                    browser = p.chromium.launch(**kwargs)
                    break
                except Exception:
                    continue
            if browser is not None:
                page = browser.new_page()
                page.goto(html_uri, wait_until="networkidle")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    prefer_css_page_size=True,
                )
                browser.close()
                return
    except Exception as exc:
        print(f"Playwright PDF fallback triggered: {exc}")

    edge_exe = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge_exe.exists():
        edge_exe = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    subprocess.run(
        [
            str(edge_exe),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            html_uri,
        ],
        check=True,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    html_file = root / "reports" / "SentinelMesh_Capstone_Design_Report.html"
    target_pdf = root / "reports" / "SentinelMesh_Capstone_Design_Report_Rohit_Raj_2301730179.pdf"
    html_to_pdf(html_file, target_pdf)
    size_mb = target_pdf.stat().st_size / (1024 * 1024)
    print(f"Generated PDF: {target_pdf} ({size_mb:.2f} MB)")
