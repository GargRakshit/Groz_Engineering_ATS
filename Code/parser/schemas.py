import re
from datetime import datetime
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator
from typing import List, Optional


_PRESENT_TOKENS = {"present", "current", "ongoing", "now", "till date", "to date", "till now"}
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_NAME_YEAR = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{4})$")
_MM_YYYY = re.compile(r"^(\d{1,2})[/-](\d{4})$")
_LEADING_BULLET = re.compile(r"^[\s*•·●▪◦\-]+")
_LOCATION_PIPE = re.compile(r"\s*\|.*$")

_MS_ALIASES: dict[str, str] = {
    "ms word": "Microsoft Word",
    "ms excel": "Microsoft Excel",
    "ms outlook": "Microsoft Outlook",
    "ms powerpoint": "Microsoft PowerPoint",
    "ms access": "Microsoft Access",
    "ms teams": "Microsoft Teams",
    "ms office": "Microsoft Office",
    "ms project": "Microsoft Project",
}


class Candidate(BaseModel):
    full_name: Optional[str] = Field(default=None, description="Candidate's full name exactly as it appears. Null only if no name appears anywhere in the resume text.")
    email: Optional[str] = Field(default=None, description="Primary email address. Use mailto: link target or visible email text. Null if absent.")
    phone: Optional[str] = Field(default=None, description="Primary phone number with country code if shown. Null if absent.")
    location: Optional[str] = Field(default=None, description="City/state/country as shown. Null if absent.")
    linkedin: Optional[str] = Field(default=None, description="LinkedIn profile URL. Null if absent.")
    github: Optional[str] = Field(default=None, description="Main GitHub profile URL. Null if absent.")
    portfolio: Optional[str] = Field(default=None, description="Personal website / portfolio / hosted profile URL. Null if absent.")


class Education(BaseModel):
    degree: Optional[str] = Field(default=None, description="Degree title verbatim, e.g. 'Bachelor of Science', 'Master of Arts : Mass Communication'.")
    field_of_study: Optional[str] = Field(default=None, description="Field/major if shown separately from the degree.")
    institution: Optional[str] = Field(default=None, description="School/university name verbatim.")
    start_date: Optional[str] = Field(default=None, description="YYYY-MM or YYYY. Null only if no start date appears.")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM or YYYY for graduation/completion. Capture year-only graduation dates like 'May 2003' → '2003-05' or '1982' → '1982'. Null only if no date appears.")
    grade: Optional[str] = Field(default=None, description="GPA, percentage, or honors text if shown.")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v):
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return None
        m = _MONTH_NAME_YEAR.match(s)
        if m:
            mon = _MONTHS.get(m.group(1)[:3].lower())
            if mon:
                return f"{m.group(2)}-{mon:02d}"
        m = _MM_YYYY.match(s)
        if m:
            return f"{m.group(2)}-{int(m.group(1)):02d}"
        return s


class Experience(BaseModel):
    company: Optional[str] = Field(default=None, description="Employer name verbatim. Use the literal text even if it is a placeholder like 'Company Name'.")
    role: Optional[str] = Field(default=None, description="Job title verbatim.")
    start_date: Optional[str] = Field(default=None, description="YYYY-MM or YYYY.")
    end_date: Optional[str] = Field(default=None, description="YYYY-MM or YYYY. MUST be null when the role is ongoing — do not put 'Current' / 'Present' here; instead set is_current=true and end_date=null.")
    is_current: bool = Field(default=False, description="True only if the resume says Present, Current, Ongoing, Now, or similar for this role.")
    description: Optional[str] = Field(
        default=None,
        description=(
            "A concise 1-3 sentence summary of this role's responsibilities and notable accomplishments. "
            "DO NOT copy every bullet verbatim — synthesize them. Mention key tools/technologies, scope "
            "(team size, budget, volume), and measurable outcomes when present. Keep under 400 characters."
        )
    )

    @field_validator("end_date", mode="before")
    @classmethod
    def _normalize_present(cls, v):
        if isinstance(v, str) and v.strip().lower() in _PRESENT_TOKENS:
            return None
        return v

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v):
        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return None
        # 'January 2012' / 'Jan 2012' / 'jan. 2012' → '2012-01'
        m = _MONTH_NAME_YEAR.match(s)
        if m:
            mon = _MONTHS.get(m.group(1)[:3].lower())
            if mon:
                return f"{m.group(2)}-{mon:02d}"
        # 'MM/YYYY' or 'M/YYYY' → 'YYYY-MM'
        m = _MM_YYYY.match(s)
        if m:
            return f"{m.group(2)}-{int(m.group(1)):02d}"
        # 'YYYY-MM' or 'YYYY' already fine
        return s


class Project(BaseModel):
    name: Optional[str] = Field(default=None, description="Project name verbatim.")
    description: Optional[str] = Field(default=None, description="Short description as written.")
    technologies_used: List[str] = Field(default_factory=list, description="Technologies/tools mentioned for this project.")
    link: Optional[str] = Field(default=None, description="Repo or demo URL. Null if none.")


class ParserMetadata(BaseModel):
    confidence_score: float = Field(default=0, ge=0, le=1, description="Confidence in the overall extraction, 0 to 1.")
    missing_important_fields: List[str] = Field(
        default_factory=list,
        description=(
            "MANDATORY: list every field below that ended up null or empty after extraction: "
            "'full_name', 'email', 'phone', 'summary', 'skills', 'experience', 'education', 'certifications'. "
            "Do not leave this empty if any of those are missing. Pure check against your own output."
        )
    )
    possible_issues: List[str] = Field(
        default_factory=list,
        description=(
            "List only issues visible in the supplied resume text (unclear dates, "
            "truncated bullets, scrambled section order). Common mojibake is already "
            "removed upstream — do not report it. Empty list is acceptable."
        )
    )
    is_messy_resume: bool = Field(default=False, description="True if text is OCR-damaged, scrambled, or hard to parse.")


def _coerce_str_list(v):
    if not isinstance(v, list):
        return v
    out = []
    for item in v:
        if item is None:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
            continue
        if isinstance(item, dict):
            label = (
                item.get("name") or item.get("title") or item.get("award")
                or item.get("certification") or item.get("value") or item.get("text")
            )
            if not label and len(item) == 1:
                label = next(iter(item.values()))
            if label:
                label = str(label).strip()
                year = item.get("year") or item.get("date")
                issuer = item.get("issuer") or item.get("organization") or item.get("org")
                extras = []
                if issuer:
                    extras.append(str(issuer).strip())
                if year:
                    extras.append(str(year).strip())
                if extras:
                    label = f"{label} ({', '.join(extras)})"
                if label:
                    out.append(label)
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


class ResumeData(BaseModel):
    candidate: Candidate
    summary: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim prose from a 'Summary', 'Executive Summary', 'Profile', 'About', 'Objective', or "
            "'Professional Summary' section. Copy the entire paragraph. Null ONLY if no such section exists."
        )
    )
    skills: List[str] = Field(
        default_factory=list,
        description=(
            "Every item from any section titled 'Skills', 'Highlights', 'Core Qualifications', 'Technical Skills', "
            "'Key Skills', 'Areas of Expertise', or 'Competencies'. Split on commas, bullets, or new lines. "
            "Preserve original casing. MUST NOT be empty if any such section exists in the resume."
        )
    )
    education: List[Education] = Field(
        default_factory=list,
        description=(
            "Every degree under 'Education', 'Education and Training', or 'Academic Background'. "
            "One entry per degree/diploma. MUST NOT be empty if such a section exists."
        )
    )
    experience: List[Experience] = Field(default_factory=list, description="Every role under 'Experience', 'Work Experience', 'Professional Experience', or 'Employment History'.")
    projects: List[Project] = Field(default_factory=list, description="Every project under 'Projects' / 'Personal Projects' / 'Side Projects'.")
    certifications: List[str] = Field(default_factory=list, description="Items under 'Certifications', 'Licenses', or certification-style entries inside 'Core Qualifications'.")
    achievements: List[str] = Field(
        default_factory=list,
        description=(
            "Every item under 'Awards', 'Achievements', 'Honors', 'Recognition', or 'Activities and Honors'. "
            "MUST NOT be empty if such a section exists."
        )
    )
    languages: List[str] = Field(default_factory=list, description="Spoken/written languages under a 'Languages' section.")
    qualifications: List[str] = Field(
        default_factory=list,
        description=(
            "3–8 word phrases describing demonstrated soft skills, domain expertise, and "
            "experience-based capabilities clearly evidenced in the resume. "
            "Examples: 'cross-functional team leadership', 'budget ownership', "
            "'regulatory compliance in financial services', 'stakeholder management', "
            "'executive-level communication'. "
            "Do NOT duplicate technical skills already captured in skills[]. "
            "Extract only what is explicitly evidenced — not inferred from job titles alone. "
            "Leave [] if no such qualifications are clearly demonstrated."
        )
    )
    parser_metadata: ParserMetadata
    parsed_at: datetime = Field(default_factory=datetime.now)

    @field_serializer("parsed_at")
    def _serialize_parsed_at(self, v: datetime) -> str:
        return v.isoformat()

    @classmethod
    def model_json_schema(cls, **kwargs):
        schema = super().model_json_schema(**kwargs)
        schema.get("properties", {}).pop("parsed_at", None)
        if "required" in schema:
            schema["required"] = [f for f in schema["required"] if f != "parsed_at"]
        return schema

    @field_validator("candidate", mode="before")
    @classmethod
    def _coerce_candidate(cls, v):
        if v is None:
            return {}
        if isinstance(v, str):
            return {"full_name": v.strip()} if v.strip() else {}
        return v

    @field_validator("parser_metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, v):
        # Guard against parser_metadata: null.
        return {} if v is None else v

    @field_validator("skills", "certifications", "achievements", "languages", "qualifications", mode="before")
    @classmethod
    def _flatten_str_lists(cls, v):
        return _coerce_str_list(v)

    @model_validator(mode="after")
    def _post_process(self):
        # 0. Null collapsed experience date ranges (end_date == start_date for non-current roles)
        for exp in self.experience:
            if (
                not exp.is_current
                and exp.start_date
                and exp.end_date
                and exp.start_date == exp.end_date
            ):
                exp.end_date = None
                label = exp.role or exp.company or "role"
                self.parser_metadata.possible_issues.append(
                    f"experience date range collapsed for {label}"
                )

        # 0b. Education: collapsed start==end means the source had only a graduation
        # date. Keep it as graduation (end_date), null start_date.
        for edu in self.education:
            if edu.start_date and edu.end_date and edu.start_date == edu.end_date:
                edu.start_date = None

        # 0c. Strip city/state location suffix (e.g. " | City , State") from
        # company and institution values. Resume PDFs often concatenate the
        # location onto the same layout line as the employer/school name.
        for exp in self.experience:
            if exp.company and "|" in exp.company:
                stripped = _LOCATION_PIPE.sub("", exp.company).strip()
                if stripped:
                    exp.company = stripped
        for edu in self.education:
            if edu.institution and "|" in edu.institution:
                stripped = _LOCATION_PIPE.sub("", edu.institution).strip()
                if stripped:
                    edu.institution = stripped

        # 1. Strip stray bullet/asterisk markers from summary
        if self.summary:
            cleaned = _LEADING_BULLET.sub("", self.summary).strip()
            # also collapse internal "*" used as sentence delimiters
            cleaned = re.sub(r"\s+\*+\s*", " ", cleaned).strip()
            self.summary = cleaned or None

        # 2. Canonicalize MS-prefix aliases then dedupe skills case-insensitively.
        # Also drop cert overlap.
        aliased = []
        for s in self.skills:
            if isinstance(s, str):
                canon = _MS_ALIASES.get(s.strip().casefold())
                aliased.append(canon if canon else s)
        self.skills = aliased

        cert_keys = {c.casefold() for c in self.certifications if isinstance(c, str)}
        seen = set()
        deduped = []
        for s in self.skills:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if len(t) < 2:
                continue
            key = t.casefold()
            if key in seen or key in cert_keys:
                continue
            seen.add(key)
            deduped.append(t)
        self.skills = deduped

        # 3. Recompute confidence_score if the LLM left it at 0 despite content
        if self.parser_metadata.confidence_score <= 0.0:
            checks = [
                bool(self.candidate and (self.candidate.full_name or self.candidate.email or self.candidate.phone)),
                bool(self.summary),
                bool(self.skills),
                bool(self.education),
                bool(self.experience),
                bool(self.certifications or self.achievements or self.projects),
            ]
            populated = sum(checks)
            if populated == 0:
                score = 0.1
            else:
                score = round(0.3 + 0.7 * (populated / len(checks)), 2)
            self.parser_metadata.confidence_score = score

        return self


class JDRequirements(BaseModel):
    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills explicitly required for the role."
    )
    preferred_skills: List[str] = Field(
        default_factory=list,
        description="Skills listed as nice-to-have, preferred, or a plus."
    )
    min_years_experience: Optional[float] = Field(
        default=None,
        description="Minimum years of relevant work experience stated. Null if not specified."
    )
    required_education_level: Optional[str] = Field(
        default=None,
        description="Lowest acceptable degree level: 'high school', 'associate', 'bachelor', 'master', or 'phd'. Null if not specified."
    )
    required_certifications: List[str] = Field(
        default_factory=list,
        description="Certifications/licenses explicitly required."
    )
    preferred_certifications: List[str] = Field(
        default_factory=list,
        description="Certifications/licenses listed as preferred or nice-to-have."
    )
    required_qualifications: List[str] = Field(
        default_factory=list,
        description=(
            "Experience-based requirements stated as required/must-have. "
            "Concise 3–8 word phrases describing soft skills, domain expertise, or leadership "
            "experience. Examples: 'cross-functional team leadership', 'budget ownership', "
            "'regulated industry experience'. Not technical skills or certifications — those go elsewhere."
        )
    )
    preferred_qualifications: List[str] = Field(
        default_factory=list,
        description=(
            "Experience-based requirements stated as preferred or nice-to-have. "
            "Same phrase format as required_qualifications."
        )
    )

    @field_validator(
        "required_skills", "preferred_skills",
        "required_certifications", "preferred_certifications",
        "required_qualifications", "preferred_qualifications",
        mode="before"
    )
    @classmethod
    def _flatten_str_lists(cls, v):
        return _coerce_str_list(v)

    @field_validator("min_years_experience", mode="before")
    @classmethod
    def _coerce_years(cls, v):
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = re.match(r"(\d+(?:\.\d+)?)", v.strip())
            if m:
                return float(m.group(1))
        return None

    @field_validator("required_education_level", mode="before")
    @classmethod
    def _normalize_edu_level(cls, v):
        if not isinstance(v, str):
            return v
        low = v.strip().lower()
        for level in ("phd", "doctorate", "master", "bachelor", "associate", "high school"):
            if level in low:
                return "phd" if level == "doctorate" else level
        return v.strip() or None
