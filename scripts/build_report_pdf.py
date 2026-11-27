"""Build publication-grade PDF report from HTML using Playwright (Edge/Chrome) or Edge CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Render an HTML document to a styled, paginated PDF."""
    html_uri = html_path.resolve().as_uri()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer Playwright with installed Microsoft Edge or Chrome channel (no extra browser download needed)
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
                    display_header_footer=True,
                    header_template=(
                        '<div style="font-size:8px;color:#64748b;width:100%;padding:0 18mm;'
                        'display:flex;justify-content:space-between;font-family:Segoe UI,Arial,sans-serif;">'
                        '<span>SentinelMesh AI — Enterprise Threat Detection &amp; Prevention Design Report</span>'
                        '<span>Meridian FinTech Inc. (Confidential Capstone)</span>'
                        '</div>'
                    ),
                    footer_template=(
                        '<div style="font-size:8px;color:#64748b;width:100%;padding:0 18mm;'
                        'display:flex;justify-content:space-between;font-family:Segoe UI,Arial,sans-serif;">'
                        '<span>AI &amp; Cybersecurity Capstone | 120 Marks Specification</span>'
                        '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>'
                        '</div>'
                    ),
                    margin={"top": "20mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                )
                browser.close()
                return
    except Exception as exc:
        print(f"Playwright PDF fallback triggered: {exc}")

    # Fallback: Microsoft Edge headless CLI
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
    html_file = root / "reports" / "SentinelMesh_AI_Capstone_Report.html"
    pdf_file = root / "reports" / "SentinelMesh_AI_Capstone_Report.pdf"
    html_to_pdf(html_file, pdf_file)
    size_mb = pdf_file.stat().st_size / (1024 * 1024)
    print(f"Generated PDF: {pdf_file} ({size_mb:.2f} MB)")
