import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

class StagingExtractor:
    def __init__(self):
        self.regex_patterns = {
            'tnm': re.compile(r'\b(T[0-4isX]{1,3}N[0-3MX]{1,3}M[0-1X]{1})\b', re.I),
            'stage': re.compile(r'\b(Stage\s+[0IVX]{1,4}[A-C]?)\b', re.I)
        }
        self.llm_model = None
        self.use_llm = False

    def _load_llm(self):
        """Lazy-load LLM model"""
        if not self.llm_model:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-r1-distill-llama-8b", 
                trust_remote_code=False
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/deepseek-r1-distill-llama-8b"
            )

    def _regex_extract(self, text: str) -> List[Dict]:
        """Initial regex-based extraction"""
        findings = []
        
        # Find TNM classifications
        for match in self.regex_patterns['tnm'].finditer(text):
            findings.append({
                'stage': match.group(1).upper(),
                'system': 'TNM',
                'evidence': text[max(0, match.start()-50):match.end()+50]
            })
        
        # Find stage classifications
        for match in self.regex_patterns['stage'].finditer(text):
            findings.append({
                'stage': match.group(1).title(),
                'system': 'General',
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
        - Evidence: [exact text snippet]"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm_model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse LLM response and combine with regex findings
        return self._parse_llm_response(response) + self._regex_extract(text)

    def extract_staging(self, text: str, context_date: str) -> List[Dict]:
        """Main extraction method"""
        if self.use_llm:
            return self._llm_extract(text)
        else:
            return self._regex_extract(text)

    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse LLM output into structured format"""
        # Implementation simplified for space
        return []  # Actual parsing would handle multiple formats 
