#%%
import anthropic
from typing import List, Dict, Set, Optional
import json
from pathlib import Path
import time
from tqdm import tqdm
import random

from keys import ANTHROPIC_API_KEY

class DomainSubjectGenerator:
    def __init__(
        self,
        api_key: str,
        subjects_per_domain: int = 200,
        output_file: str = "domain_subjects.json"
    ):
        self.client = anthropic.Client(api_key=api_key)
        self.subjects_per_domain = subjects_per_domain
        self.output_file = Path(output_file)
        
        # Define domains and their descriptions
        self.domains = {
            "historical_events": "significant historical events, periods, and movements",
            "musicians": "famous musicians, bands, and composers across different genres and eras",
            "animals": "animals from different species, habitats, and categories",
            "plants": "plants, trees, flowers, and other botanical subjects",
            "abstract_concepts": "philosophical, mathematical, and abstract concepts",
            "emotions": "human emotions, feelings, and psychological states",
            "technology": "technologies, inventions, and digital concepts",
            "places": "cities, countries, landmarks, and geographical features",
            "art": "art forms, movements, and artistic concepts",
            "science": "scientific concepts, theories, and phenomena",
            "sports": "sports, games, and athletic activities",
            "food": "foods, dishes, and culinary concepts",
            "literature": "literary works, genres, and writing concepts",
            "professions": "jobs, careers, and professional roles",
            "weather": "weather phenomena and atmospheric conditions",
            "architecture": "architectural styles, structures, and building types",
            "fashion": "clothing, fashion items, and style concepts",
            "transportation": "vehicles, transport methods, and travel concepts",
            "mythology": "mythological creatures, deities, and legends",
            "human_body": "parts of the human body and biological processes",
            "languages": "languages, dialects, and linguistic concepts",
            "chemical_elements": "elements, compounds, and chemical processes",
            "geological_formations": "natural earth formations and geological features",
            "space_objects": "celestial bodies, astronomical phenomena, and cosmic objects",
            "marine_life": "sea creatures, oceanic phenomena, and underwater features",
            "dance_forms": "dance styles, movements, and choreographic concepts",
            "musical_instruments": "instruments, sound-making devices, and musical equipment",
            "weapons": "historical and modern weapons, armor, and military equipment",
            "tools": "tools, implements, and devices for various tasks",
            "movies": "films, cinema concepts, and movie-related terminology",
            "games": "board games, card games, and recreational activities",
            "religions": "religious beliefs, practices, and spiritual concepts",
            "materials": "natural and synthetic materials, substances, and compounds",
            "colors": "colors, shades, hues, and color-related concepts",
            "insects": "insects, arachnids, and other small creatures",
            "television": "TV shows, broadcasting concepts, and television terminology",
            "diseases": "medical conditions, ailments, and health disorders",
            "festivals": "celebrations, holidays, and cultural events",
            "minerals": "minerals, gems, precious stones, and geological resources",
            "social_media": "social platforms, online communication, and digital social concepts",
            "furniture": "furniture items, home furnishings, and interior objects",
            "birds": "bird species, avian behaviors, and flight-related concepts",
            "crimes": "types of crimes, criminal behavior, and legal violations",
            "drinks": "beverages, drink types, and liquid refreshments",
            "time_periods": "eras, epochs, and historical time divisions",
            "sports_equipment": "athletic gear, sporting goods, and fitness equipment",
            "natural_disasters": "catastrophic events, environmental phenomena, and disasters",
            "hobbies": "recreational activities, pastimes, and leisure pursuits",
            "advertising": "marketing concepts, promotional methods, and advertising terminology",
            "phobias": "fears, anxieties, and psychological aversions"
        }

    def generate_prompt(self, domain: str, description: str) -> str:
        return f"""Generate a list of exactly {self.subjects_per_domain} subjects related to {domain} ({description}).

Requirements:
- One subject per line
- Only the subject should appear on each line: no numbers, prefixes, suffixes, etc
- Keep each subject concise (1-4 words)
- Focus on commonly known and understood subjects
- Avoid obscure or highly technical terms
- No duplicate concepts
- Mix specific and general concepts
- Include subjects from different cultures and time periods where applicable

Please provide exactly {self.subjects_per_domain} subjects, one per line. Print only the subjects, with no numbering."""

    def get_domain_subjects(self, domain: str, description: str) -> List[str]:
        """Get subjects for a specific domain using Claude Haiku."""
        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": self.generate_prompt(domain, description)
                }]
            )
            
            # Parse response into list of subjects
            subjects = [
                line.strip() for line in message.content[0].text.split('\n')
                if line.strip() and not line.startswith('-')
            ]
            
            # Ensure we get exactly the number we want
            if len(subjects) > self.subjects_per_domain:
                subjects = subjects[:self.subjects_per_domain]
            
            return subjects
            
        except Exception as e:
            print(f"Error generating subjects for {domain}: {str(e)}")
            return []

    def load_existing_subjects(self) -> Dict[str, List[str]]:
        """Load existing subjects if file exists."""
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                return json.load(f)
        return {}

    def save_subjects(self, subjects_dict: Dict[str, List[str]]):
        """Save subjects to file."""
        with open(self.output_file, 'w') as f:
            json.dump(subjects_dict, f, indent=2)

    def generate_all_subjects(self, delay: float = 0.5) -> Dict[str, List[str]]:
        """Generate subjects for all domains with progress tracking."""
        all_subjects = self.load_existing_subjects()
        domains_to_process = {k: v for k, v in self.domains.items() if k not in all_subjects}
        
        if not domains_to_process:
            print("All domains already processed!")
            return all_subjects
            
        print(f"Generating subjects for {len(domains_to_process)} domains...")
        
        for domain, description in tqdm(domains_to_process.items()):
            print(f"\nProcessing domain: {domain}")
            subjects = self.get_domain_subjects(domain, description)
            
            if subjects:
                all_subjects[domain] = subjects
                print(f"Generated {len(subjects)} subjects for {domain}")
                
                # Save progress after each domain
                self.save_subjects(all_subjects)
                
                # Rate limiting
                time.sleep(delay)
            else:
                print(f"Failed to generate subjects for {domain}")
        
        return all_subjects

    def analyze_subjects(self, subjects_dict: Dict[str, List[str]]):
        """Analyze the generated subjects."""
        analysis = {
            "total_domains": len(subjects_dict),
            "total_subjects": sum(len(subjects) for subjects in subjects_dict.values()),
            "subjects_per_domain": {
                domain: len(subjects) for domain, subjects in subjects_dict.items()
            },
            "word_counts": {
                domain: sum(len(s.split()) for s in subjects)
                for domain, subjects in subjects_dict.items()
            }
        }
        
        # Check for potential duplicates across domains
        all_subjects = [s.lower() for subjects in subjects_dict.values() for s in subjects]
        duplicates = {s for s in all_subjects if all_subjects.count(s) > 1}
        analysis["duplicate_count"] = len(duplicates)
        
        return analysis

    def get_random_sample(self, subjects_dict: Dict[str, List[str]], n: int = 5) -> Dict[str, List[str]]:
        """Get random sample of subjects from each domain."""
        return {
            domain: random.sample(subjects, min(n, len(subjects)))
            for domain, subjects in subjects_dict.items()
        }
    


class DatasetGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1000,
        min_tokens: Optional[int] = None,
        temperature: float = 1.0,
        domain_subjects_file: str = "domain_subjects.json",
        output_file: str = "responses.json"
    ):
        self.client = anthropic.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.domain_subjects_file = Path(domain_subjects_file)
        self.output_file = Path(output_file)

    def load_domain_subjects(self) -> List[str]:
        """Load and flatten domain subjects into a single list."""
        if not self.domain_subjects_file.exists():
            raise FileNotFoundError(f"Domain subjects file not found: {self.domain_subjects_file}")
            
        with open(self.domain_subjects_file, 'r') as f:
            domain_subjects = json.load(f)
            
        # Flatten all subjects from all domains into a single list
        all_subjects = [
            subject for subjects in domain_subjects.values()
            for subject in subjects
        ]
        
        print(f"Loaded {len(all_subjects)} subjects from {len(domain_subjects)} domains")
        return all_subjects

    def collect_responses(
        self,
        prompt_template: str,
        delay: float = 0.0
    ) -> List[Dict[str, str]]:
        """Collect responses for each subject using the provided prompt template."""
        subjects = self.load_domain_subjects()
        dataset = []
        existing_responses = {}

        # Load existing responses if file exists
        if self.output_file.exists():
            with open(self.output_file, 'r') as f:
                existing_data = json.load(f)
                existing_responses = {item["subject"]: item["preferred"] for item in existing_data}
                print(f"Loaded {len(existing_responses)} existing responses")

        # Create progress bar
        pbar = tqdm(subjects, desc="Collecting responses")
        
        for subject in pbar:
            pbar.set_description(f"Processing: {subject[:30]}...")
            
            if subject in existing_responses:
                dataset.append({
                    "subject": subject,
                    "preferred": existing_responses[subject]
                })
                continue

            prompt = prompt_template.format(subject=subject)
            
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response = message.content[0].text.strip()
                
                # Check if response meets minimum token requirement
                if self.min_tokens and len(response.split()) < self.min_tokens:
                    pbar.write(f"Response for '{subject}' too short, retrying...")
                    continue

                dataset.append({
                    "subject": subject,
                    "preferred": response
                })

                # Save after each successful response
                with open(self.output_file, 'w') as f:
                    json.dump(dataset, f, indent=2)

                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                pbar.write(f"Error collecting response for '{subject}': {str(e)}")
                continue

        print(f"\nCompleted! Generated {len(dataset)} responses")
        return dataset
    



#%% Generate list of subjects
generator = DomainSubjectGenerator(
    api_key=ANTHROPIC_API_KEY,
    subjects_per_domain=100
)
subjects_dict = generator.generate_all_subjects(delay=0.0)




#%% Generate responses for each subject in a chosen style (e.g. poetry)
generator = DatasetGenerator(
    api_key=ANTHROPIC_API_KEY,
    max_tokens=70,
    min_tokens=20,
    temperature=0.7
)
template = "Please write a short poem about the following subject: {subject}. Do not introduce the poem. Only respond with the poem itself. You may only speak in rhyming verse."

dataset = generator.collect_responses(
    prompt_template=template,
    delay=0.0  # Adjust if needed for rate limiting
)