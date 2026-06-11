import json


def build_resume_extraction_prompt(text, extracted_links):
    return f"""
You are a resume parsing engine. Extract structured information from the resume text below.

Output a single JSON object matching the provided schema. No markdown, no commentary.
The top-level object MUST have these keys directly: 'candidate', 'summary', 'skills',
'experience', 'education', 'projects', 'certifications', 'achievements', 'languages',
'qualifications', 'parser_metadata'. Do NOT wrap the output in any envelope key like 'resume' or 'data'.

# Rules
1. Extract only what is present in the resume text. Do not infer or invent.
2. Use null for missing single-value fields, [] for missing list fields.
3. Preserve original wording for names, titles, degrees, companies, and skills.
4. Anonymized placeholders like 'Company Name', 'City', 'State' are part of the source — copy them verbatim.
5. Normalize dates to YYYY-MM (e.g. 'May 2003' → '2003-05', '01/2016' → '2016-01').
   Year-only → YYYY. For ongoing roles set is_current=true and end_date=null.
6. Replace any mojibake (Â, â€™, â€œ, â€, â€¦, ï¼, zero-width chars) with the intended ASCII
   equivalent or drop it. The final JSON must contain only clean text.
7. Only populate the name field if there is a human name in the resume that refers to the applicant.

# Candidate / Contact information
- full_name: the applicant's own name. Typically the largest / most prominent text on
  page 1, or follows a "Name:" / "Candidate:" label. Apply rule 7.
- email: look for labels "EMAIL", "E-MAIL", "EMAIL ID", "E-MAIL ID", "Mail", "E:" and
  copy the address that follows. Also check extracted_links for any mailto: entry and
  extract the address from it. Copy verbatim (preserve case). null if absent.
- phone: look for labels "PHONE", "MOBILE", "MOB", "MOB.", "CELL", "CONTACT",
  "TEL", "PH", "M:", "Ph:" and copy the number that follows, including country code
  (+91, +1, etc.). Also match standalone numeric sequences that look like phone numbers
  (10+ digits, often with spaces or dashes). Copy verbatim. null if absent.
- location: the applicant's personal address or city/region from the contact header
  section (NOT from experience entries). Look for "ADDRESS", "LOCATION", "CITY",
  "RESIDENCE", "DOMICILE" labels, or an address block that appears directly under the
  name. Extract the most useful portion (city, state/district, country). null if absent.
- linkedin: extract from extracted_links or text when a linkedin.com URL appears.
- github: extract from extracted_links or text when a github.com URL appears.
- portfolio: extract from extracted_links or text for any other personal/portfolio URL.

# Sections → fields
- 'Summary' / 'Profile' / 'Objective' / 'About'        → summary (verbatim paragraph)
- 'Skills' / 'KEY SKILL' / 'KEY SKILLS' / 'Highlights' / 'Core Qualifications' /
  'Technical Skills' / 'Areas of Expertise'             → skills (one entry per item).
  IMPORTANT: read the ENTIRE document — some resumes have both a 'Highlights' section
  near the top AND a separate 'Skills' section near the bottom. Extract from BOTH and
  merge into a single deduplicated list.
  ALSO scan every experience/responsibility bullet for named domain technologies, machine
  types (e.g. "PDC", "die casting machine", "CNC"), industry-specific processes (e.g.
  "pressure die casting", "die setting", "process parameter setting"), and professional
  tools or standards (e.g. "IATF 16949", "SAP", "ERP"). Add these as additional skill
  entries even if they do not appear under a formal Skills section heading. Only extract
  named, specific things — skip generic verbs and adjectives.
- 'Education' / 'Academic Background'                   → education.
  Education entries often use 'Degree : Field' colon syntax — extract text after the
  colon as field_of_study. E.g. 'Master of Arts : Mass Communication' → degree='Master
  of Arts', field_of_study='Mass Communication'.
  If only a graduation date appears (e.g. 'May 2003'), set start_date=null and
  end_date=that date. If NO date appears at all for an entry, BOTH start_date AND
  end_date MUST be null — never guess, never use today's date.
- 'Experience' / 'Work Experience' /
  'Professional Experience' / 'Employment History'      → experience
- 'Projects' / 'Personal Projects'                      → projects
- 'Certifications' / 'Licenses'                         → certifications
- 'Awards' / 'Achievements' / 'Honors' / 'Recognition'  → achievements
- 'Languages'                                           → languages

# Experience.description
Write a concise 1–3 sentence SUMMARY of each role. Do not copy every bullet verbatim.
Mention key responsibilities, tools, scope (team size, budget, volume), and outcomes only
when they appear in the source. Keep under 400 characters per role.

# Experience entry rules
- company: the employer name only. Strip city/state location suffixes — if the source
  reads 'Company Name | City , State' or 'Employer City , State [role]', extract only
  'Company Name' or 'Employer'. Never include '|', city, or state in the company field.
  This stripping rule applies to experience.company ONLY — NOT to education.institution.
  Always extract the full school/university name into institution verbatim.
- role: the job title for the block. Look at the line(s) containing or adjacent to
  the date range and pick the text that is NOT the company name and NOT the date.
  Titles can appear before the date ("Public Relations Director , 01/2016 to Current"),
  after the date ("Company Name City , State HR Personnel Assistant 03/2013 to 04/2014"),
  or on a separate heading line. Copy verbatim — keep slash-joined titles as one
  string ("HR/Payroll Supervisor Accounting Apprentice"). Set role=null ONLY if no
  title text appears anywhere in the block. Do NOT infer role from bullet content.
- Every entry MUST be anchored to an explicit date or date range in the source.
  If a block lists roles WITHOUT any date, skip it entirely — skip both its role text
  AND its description bullets. Do NOT merge them into the adjacent dated entry below.
  The dated entry that immediately follows an undated block is its own independent record;
  extract its role and description ONLY from its own content, ignoring the undated block.
  Never invent dates.

# Certifications vs skills
Items like 'PMP', 'SOX Training', 'CPR/AED Certification', 'Six Sigma' are certifications,
not skills — put them in `certifications` even if they appear under a Skills heading.

# Skill entry rules
- If a "skill" is a parenthesized comma list of tools (e.g. "HRIS systems (Banner,
  PeopleAdmin, PMIS, BES, VNAV)"), split the parenthesized items into separate
  skill entries. You may keep the umbrella label too if it adds meaning.
- Emit only named tools, technologies, methodologies, or domain competencies
  (e.g. "Accounts payable", "Financial forecasting", "PeopleSoft", "SOX compliance").
  Omit bare generic verbs/adjectives even when present in the source skills list:
  "managing", "reporting", "concise", "delivery", "type", "receiving", "policies",
  "billings", "administrative", "proposals", "Mail", "Office", "managing", "Copy".
- Only include a skill if it isn't a basic thing any person can do and is something
  that a recruiter might look for in a resume. Specific non-general skills only,
  no random adjectives or nouns. No places or things.

# Qualifications
Extract 3–8 word phrases for the `qualifications` field describing what the candidate has
demonstrably done or shown — domain expertise, leadership, operational ownership, and
experience-based capabilities. Ground every phrase in visible resume evidence (job
descriptions, achievements, summary, roles). Do NOT invent. Do NOT repeat technical skills
already in skills[].
Source ONLY from: experience/responsibility bullets, summary, and achievements.
Do NOT derive qualifications by rephrasing items already in skills[].
Look for: scope of ownership ("overall responsibility for X department"), operational
leadership ("led Y-person team", "managed production line"), customer-facing work
("handled OEM customer complaints"), compliance ownership ("maintained IATF 16949").
White-collar examples: 'cross-functional team leadership', 'P&L ownership'.
Manufacturing examples: 'PDC department operational leadership', 'OEM customer complaint
resolution', 'preventive maintenance programme ownership', 'die trial and stabilization'.
Do NOT leave [] just because the resume lacks a formal qualifications section — extract
from experience descriptions if evidence is present there.

# parser_metadata (self-check on your own output)
- missing_important_fields: list every one of 'full_name', 'email', 'phone', 'summary',
  'skills', 'experience', 'education', 'certifications' that ended up null or [].
- possible_issues: list ONLY issues actually visible in the resume text above. The
  text has already been cleaned of common mojibake (Â, â€™, â€œ, ï¼, zero-widths)
  upstream — do NOT report these as "removed" unless you literally see those
  characters in the input. Empty list is acceptable.
- is_messy_resume: true if the source text is OCR-damaged or scrambled.
- confidence_score: float in (0.0, 1.0]. Never 0.0 if you extracted any content.

# Inputs

Resume text:
\"\"\"
{text}
\"\"\"

Extracted document links:
{json.dumps(extracted_links, indent=2)}
""".strip()


def build_jd_extraction_prompt(text: str) -> str:
    return f"""
You are a job description parsing engine. Extract structured hiring requirements from the job description below.

Output a single JSON object with exactly these top-level keys: 'required_skills', 'preferred_skills', 'min_years_experience', 'required_education_level', 'required_certifications', 'preferred_certifications', 'required_qualifications', 'preferred_qualifications'. No markdown, no commentary, no wrapping keys.

# Rules
1. Extract only what is stated. Do not infer, invent, or embellish.
2. Use [] for missing list fields, null for missing scalar fields.
3. required vs preferred: use the JD's own language — "required", "must", "essential" → required_*; "preferred", "nice to have", "a plus", "desired", "ideally" → preferred_*. If the JD gives a single undifferentiated list, put everything in required_*.
4. min_years_experience: extract the minimum number stated (e.g. "3–5 years" → 3.0, "at least 2 years" → 2.0, "2+ years" → 2.0). Null if not mentioned.
5. required_education_level: normalize to one of exactly: "high school", "associate", "bachelor", "master", "phd". Null if not specified.
6. skills: named tools, technologies, and methodologies only (Python, AWS, Agile, SQL, Excel). Do NOT include vague experience statements — those are not extractable as skills.
7. certifications: named credentials only (PMP, AWS Solutions Architect, CPA, Six Sigma Green Belt). Do NOT place certifications in skills.
8. qualifications: experience-based requirements that are NOT named skills or certifications. Concise 3–8 word phrases only. Examples: 'cross-functional team leadership', 'regulated industry experience', 'executive stakeholder management', 'P&L ownership'. Apply required/preferred split the same as other fields.

# Job description text:
\"\"\"
{text}
\"\"\"
""".strip()
