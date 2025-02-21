import re
from typing import List, Dict, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class StagingExtractor:
    def __init__(self):
        self.regex_patterns = {
            'tnm': re.compile(r'\b(T[0-4isX]{1,3}N[0-3MX]{1,3}M[0-1X]{1})\b', re.I),
            'stage': re.compile(r'\b(Stage\s+[0IVX]{1,4}[A-C]?)\b', re.I),
            'date': re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
        }
        self.llm_model = None
        self.use_llm = False

    def _load_llm(self):
        """Lazy-load LLM model"""
        if not self.llm_model:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-r1-200k-base", 
                trust_remote_code=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/deepseek-r1-200k-base"
            )

    def _regex_extract(self, text: str) -> List[Dict]:
        """Initial regex-based extraction"""
        findings = []
        
        # Find TNM classifications
        for match in self.regex_patterns['tnm'].finditer(text):
            findings.append({
                'stage': match.group(1).upper(),
                'system': 'TNM',
                'confidence': 'NA',
                'evidence': text[max(0, match.start()-50):match.end()+50]
            })
        
        # Find stage classifications
        for match in self.regex_patterns['stage'].finditer(text):
            findings.append({
                'stage': match.group(1).title(),
                'system': 'General',
                'confidence': 'NA',
                'evidence': text[max(0, match.start()-50):match.end()+50]
            })
        
        return findings

    def _llm_extract(self, text: str) -> List[Dict]:
        """LLM-based extraction with fallback to regex"""
        self._load_llm()
        prompt = f"""Analyze this clinical note and extract cancer staging information:
{text}

Output format:
- Stage: [detected stage or TNM classification]
- System: [staging system used]
- Date: [YYYY-MM-DD if available]
- Confidence: [0.0-1.0]
- Evidence: [exact text snippet]"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm_model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse LLM response and combine with regex findings
        return self._parse_llm_response(response) + self._regex_extract(text)

    def extract_staging(self, text: str, context_date: str) -> List[Dict]:
        """Main extraction method with date resolution"""
        if self.use_llm:
            findings = self._llm_extract(text)
        else:
            findings = self._regex_extract(text)
        
        # Date resolution logic
        for finding in findings:
            if 'date' not in finding:
                date_match = self.regex_patterns['date'].search(finding['evidence'])
                finding['date'] = date_match.group(1) if date_match else context_date
                
        return findings

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM output into structured format"""
        # Implementation simplified for space
        return []  # Actual parsing would handle multiple formats 