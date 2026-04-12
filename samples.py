"""
samples.py
----------
Ships three realistic sample documents so a first-time visitor can evaluate
InferLens in under 30 seconds without uploading their own PDF.

Each sample carries:
    * A title and short one-line description
    * A scenario framing the use case for the buyer
    * The full document text (multi-page)
    * Three suggested questions a real evaluator would ask

PDFs are generated on first use via PyMuPDF (already a dep) so no sample
binaries need to live in the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF 1.24+ — preferred import name
except ImportError:
    import fitz  # Legacy fallback for older PyMuPDF


@dataclass
class SampleDoc:
    slug: str
    title: str
    scenario: str
    description: str
    pages: list[str]
    suggested_questions: list[str]


SAMPLES_DIR = Path(__file__).parent / "sample_cache"


SAMPLES: list[SampleDoc] = [
    SampleDoc(
        slug="bank-10k",
        title="Regional Bank Annual Report (10-K excerpt)",
        scenario="You're a compliance analyst at a financial regulator. You need to extract specific risk-factor and capital-reserve language from a 10-K filing without reading all 94 pages.",
        description="Finance · Regulatory filing",
        pages=[
            "NORTHBRIDGE REGIONAL BANCORP INC.\nAnnual Report on Form 10-K\nFiscal Year Ended December 31, 2025\n\nItem 1. Business Overview\n\nNorthbridge Regional Bancorp Inc. is a bank holding company headquartered in Portland, Oregon, operating 142 branches across Oregon, Washington, and northern California. The Company's principal subsidiary, Northbridge Regional Bank, N.A., is a nationally chartered commercial bank providing retail banking, commercial lending, and wealth management services to approximately 1.2 million customer accounts. As of December 31, 2025, total assets were $38.4 billion, total deposits were $31.2 billion, and total stockholders' equity was $3.9 billion.",

            "Item 1A. Risk Factors\n\nOur business is subject to the following material risks, any of which could have a material adverse effect on our financial condition, results of operations, or cash flows.\n\nCredit Risk. Approximately 62% of our loan portfolio consists of commercial real estate loans concentrated in the Pacific Northwest. A sustained downturn in regional commercial property values could materially increase loan loss provisions. During fiscal year 2025, non-performing loans increased from 0.41% to 0.68% of total loans, driven principally by office-sector exposure in Portland and Seattle.\n\nInterest Rate Risk. Our net interest margin is sensitive to changes in short-term rates. A 100 basis point parallel downward shift in the yield curve is estimated to reduce net interest income by approximately $142 million over a twelve-month horizon.\n\nLiquidity Risk. The Company maintains a liquidity coverage ratio (LCR) of 127%, above the regulatory minimum of 100%. However, a rapid deposit outflow event could stress this ratio.",

            "Item 7. Capital Resources and Regulatory Requirements\n\nThe Company and the Bank are subject to the capital adequacy requirements of the Federal Reserve and the Office of the Comptroller of the Currency. Under the Basel III framework, the Bank is required to maintain the following minimum capital ratios:\n\n  - Common Equity Tier 1 (CET1) capital ratio: 4.5%\n  - Tier 1 capital ratio: 6.0%\n  - Total capital ratio: 8.0%\n  - Tier 1 leverage ratio: 4.0%\n\nIn addition, the Bank must maintain a 2.5% capital conservation buffer above these minimums, resulting in effective minimum ratios of 7.0% CET1, 8.5% Tier 1, and 10.5% Total capital. As of December 31, 2025, the Bank's CET1 capital ratio was 12.8%, Tier 1 ratio was 13.4%, and Total capital ratio was 15.1%, all significantly in excess of regulatory minimums and the well-capitalized thresholds.",

            "Item 8. Deposit Insurance and Liquidity\n\nDeposits at Northbridge Regional Bank, N.A. are insured by the Federal Deposit Insurance Corporation up to applicable limits. As of December 31, 2025, approximately 71% of total deposits were fully insured, with the remaining 29% representing uninsured deposits principally from commercial customers. Management monitors uninsured deposit concentration weekly and maintains contingent funding facilities totaling $8.2 billion, including $5.1 billion in Federal Home Loan Bank advances and $3.1 billion at the Federal Reserve discount window.",
        ],
        suggested_questions=[
            "What are the Tier 1 capital requirements and how does Northbridge compare?",
            "What is the Company's exposure to commercial real estate and how has it changed?",
            "How much of Northbridge's deposit base is uninsured?",
        ],
    ),

    SampleDoc(
        slug="drug-label",
        title="Prescription Drug Label — Cardiovascular Medication",
        scenario="You're a clinical pharmacist reviewing a new formulary submission. You need to verify dosing, contraindications, and drug interactions without scanning the entire label.",
        description="Healthcare · Regulated product information",
        pages=[
            "VELOXAR (altirostatin sodium) Tablets\nFor Oral Use\n\n1. INDICATIONS AND USAGE\n\nVELOXAR is indicated as an adjunct to diet to reduce low-density lipoprotein cholesterol (LDL-C) in adult patients with primary hyperlipidemia, and to reduce the risk of cardiovascular events in adults with established atherosclerotic cardiovascular disease. VELOXAR has not been studied in patients with Fredrickson Type I or V dyslipidemia.",

            "2. DOSAGE AND ADMINISTRATION\n\nThe recommended starting dose of VELOXAR is 20 mg orally once daily, with or without food. The dose may be titrated at intervals of not less than 4 weeks to a maximum dose of 80 mg once daily based on LDL-C response and tolerability. For patients with moderate hepatic impairment (Child-Pugh B), the recommended starting dose is 10 mg once daily and the maximum dose should not exceed 40 mg. VELOXAR is contraindicated in patients with severe hepatic impairment (Child-Pugh C).\n\nTablets should be swallowed whole with water. Do not crush, split, or chew.",

            "3. CONTRAINDICATIONS\n\nVELOXAR is contraindicated in the following patient populations:\n  - Known hypersensitivity to altirostatin or any excipient\n  - Active liver disease or unexplained persistent elevations of serum transaminases greater than 3 times the upper limit of normal\n  - Pregnancy and breastfeeding (VELOXAR is Category X)\n  - Concomitant use with strong CYP3A4 inhibitors including itraconazole, ketoconazole, posaconazole, clarithromycin, ritonavir, and grapefruit juice consumption exceeding 1 liter per day\n  - Severe hepatic impairment (Child-Pugh C)",

            "4. WARNINGS AND PRECAUTIONS\n\n4.1 Skeletal Muscle Effects. Cases of myopathy and rhabdomyolysis with acute renal failure secondary to myoglobinuria have been reported with VELOXAR. Risk is dose-dependent and increases in patients with advanced age (>65), uncontrolled hypothyroidism, renal impairment, and concomitant gemfibrozil use. Discontinue VELOXAR if creatine kinase levels exceed 10 times the upper limit of normal or if myopathy is diagnosed or suspected.\n\n4.2 Hepatic Effects. Monitor liver enzymes before initiating therapy and as clinically indicated thereafter. Discontinue VELOXAR if ALT or AST elevations greater than 3 times ULN persist.\n\n4.3 Drug Interactions. Co-administration with cyclosporine increases altirostatin exposure 4 to 7-fold and is not recommended. Co-administration with gemfibrozil should be avoided due to increased risk of myopathy. Warfarin INR should be monitored more frequently when VELOXAR is initiated or discontinued.",
        ],
        suggested_questions=[
            "What is the maximum dose of VELOXAR for a patient with moderate hepatic impairment?",
            "Which CYP3A4 inhibitors are contraindicated with VELOXAR?",
            "What should I monitor if a patient on VELOXAR is also taking warfarin?",
        ],
    ),

    SampleDoc(
        slug="saas-msa",
        title="Enterprise SaaS Master Services Agreement",
        scenario="You're legal counsel reviewing a vendor contract. You need to quickly locate liability caps, termination rights, and data-protection clauses without reading all 40 pages.",
        description="Legal · Contract review",
        pages=[
            "MASTER SERVICES AGREEMENT\n\nThis Master Services Agreement (the 'Agreement') is entered into as of January 15, 2026 (the 'Effective Date') by and between Atlas Data Systems Inc., a Delaware corporation with its principal place of business at 500 California Street, Suite 1200, San Francisco, California 94104 ('Vendor'), and the Customer identified on the applicable Order Form.\n\n1. SERVICES\n\nVendor shall provide the software-as-a-service platform and related services described in each mutually executed Order Form (the 'Services'). Vendor shall use commercially reasonable efforts to make the Services available 24 hours a day, 7 days a week, except for planned maintenance windows not to exceed 4 hours per calendar month, scheduled during off-peak hours with at least 48 hours advance notice.",

            "2. FEES AND PAYMENT\n\n2.1 Customer shall pay all fees specified in the Order Form. Unless otherwise stated, fees are invoiced annually in advance and payable net 30 days from the invoice date.\n\n2.2 Late payments shall accrue interest at the lesser of 1.5% per month or the maximum rate permitted by law.\n\n2.3 All fees are exclusive of taxes, and Customer is responsible for all applicable sales, use, VAT, and withholding taxes.\n\n3. TERM AND TERMINATION\n\n3.1 The initial subscription term shall be as specified in the Order Form and shall automatically renew for successive one-year periods unless either party provides written notice of non-renewal at least 60 days prior to the end of the then-current term.\n\n3.2 Either party may terminate this Agreement for cause upon 30 days written notice of a material breach if such breach remains uncured at the end of the notice period. Vendor may suspend Services immediately upon non-payment of undisputed fees that remain overdue for more than 15 days.",

            "4. LIMITATION OF LIABILITY\n\n4.1 EXCEPT FOR BREACHES OF CONFIDENTIALITY, INDEMNIFICATION OBLIGATIONS, OR CUSTOMER'S PAYMENT OBLIGATIONS, IN NO EVENT SHALL EITHER PARTY'S AGGREGATE LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE TOTAL FEES PAID OR PAYABLE BY CUSTOMER TO VENDOR IN THE TWELVE (12) MONTHS PRECEDING THE EVENT GIVING RISE TO THE CLAIM.\n\n4.2 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, OR PUNITIVE DAMAGES, INCLUDING LOST PROFITS OR LOST DATA, REGARDLESS OF THE THEORY OF LIABILITY.",

            "5. DATA PROTECTION AND SECURITY\n\n5.1 Vendor shall maintain a comprehensive information security program that includes administrative, physical, and technical safeguards designed to protect the confidentiality, integrity, and availability of Customer Data. Vendor shall maintain SOC 2 Type II attestation and ISO 27001 certification throughout the term of this Agreement.\n\n5.2 Customer Data shall be processed only for the purposes of providing the Services and shall not be used for training machine learning models or sold to third parties. Vendor shall notify Customer of any confirmed Security Incident within 48 hours of discovery and shall cooperate with Customer's investigation and remediation efforts.\n\n5.3 Upon termination, Vendor shall, upon Customer's written request made within 30 days of the termination date, return or delete all Customer Data in its possession.",
        ],
        suggested_questions=[
            "What is the liability cap under this agreement and what are the carve-outs?",
            "How much notice is required to terminate without cause?",
            "What security certifications must the Vendor maintain?",
        ],
    ),
]


def _build_pdf_bytes(sample: SampleDoc) -> bytes:
    """Render a SampleDoc into PDF bytes using PyMuPDF."""
    doc = fitz.open()
    for page_text in sample.pages:
        page = doc.new_page(width=612, height=792)  # US Letter
        rect = fitz.Rect(72, 72, 540, 720)
        page.insert_textbox(
            rect,
            page_text,
            fontsize=11,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT,
        )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def get_sample_pdf_bytes(slug: str) -> bytes:
    """Return PDF bytes for the sample with the given slug.

    Caches the rendered PDF to disk (sample_cache/) so repeat loads are
    instant and we don't re-render on every rerun.
    """
    sample = next((s for s in SAMPLES if s.slug == slug), None)
    if sample is None:
        raise ValueError(f"Unknown sample slug: {slug}")

    SAMPLES_DIR.mkdir(exist_ok=True)
    cache_path = SAMPLES_DIR / f"{slug}.pdf"
    if cache_path.exists():
        return cache_path.read_bytes()

    pdf_bytes = _build_pdf_bytes(sample)
    cache_path.write_bytes(pdf_bytes)
    return pdf_bytes


def get_sample(slug: str) -> SampleDoc:
    sample = next((s for s in SAMPLES if s.slug == slug), None)
    if sample is None:
        raise ValueError(f"Unknown sample slug: {slug}")
    return sample
