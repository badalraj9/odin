"""
ODIN Scenario Generator (Enhanced)
-----------------------------------
Generates diverse, realistic training scenarios for cognitive reasoning.
"""

import random
import hashlib
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from schema import (
    Scenario, ScenarioStep, InputState, ExpectedOutput, ScenarioMetadata,
    ScenarioType, Stance, Action
)


# =============================================================================
# EXPANDED DOMAIN TEMPLATES (10+ domains)
# =============================================================================

DOMAINS = {
    # Software Development
    "backend": {
        "intents": [
            "Build a REST API for {entity} management",
            "Create CRUD endpoints for {entity}",
            "Add authentication to the {service} service",
            "Fix the performance issue in the {feature} handler",
            "Implement rate limiting for the {endpoint} endpoint",
            "Add database migrations for the {table} table",
            "Set up Redis caching for {feature}",
            "Create a webhook handler for {integration}",
            "Implement pagination for the {entity} list endpoint",
            "Add error handling to the {service} service",
        ],
        "fills": {
            "entity": ["user", "product", "order", "payment", "subscription", "invoice", "customer", "inventory"],
            "service": ["auth", "billing", "notification", "analytics", "search", "recommendation"],
            "feature": ["login", "checkout", "search", "upload", "export", "sync"],
            "endpoint": ["public", "admin", "webhook", "batch"],
            "table": ["users", "orders", "products", "transactions", "logs"],
            "integration": ["Stripe", "SendGrid", "Twilio", "Slack", "GitHub"],
        },
    },
    "frontend": {
        "intents": [
            "Create a {component} component in React",
            "Build a responsive {page} page",
            "Add dark mode support to the {area} area",
            "Implement lazy loading for {feature}",
            "Fix the CSS layout issue in {component}",
            "Add form validation to the {form} form",
            "Create a reusable {element} component",
            "Implement drag and drop for {feature}",
            "Add accessibility features to {component}",
            "Optimize bundle size for the {module} module",
        ],
        "fills": {
            "component": ["Dashboard", "Settings", "Profile", "Sidebar", "Modal", "DataTable", "Chart"],
            "page": ["landing", "pricing", "about", "contact", "dashboard", "onboarding"],
            "area": ["header", "sidebar", "main content", "footer"],
            "feature": ["images", "comments", "feed", "notifications"],
            "form": ["login", "registration", "checkout", "contact", "settings"],
            "element": ["Button", "Input", "Card", "Dropdown", "Toast"],
            "module": ["admin", "public", "shared"],
        },
    },
    "devops": {
        "intents": [
            "Set up CI/CD pipeline for {project}",
            "Configure Kubernetes deployment for {service}",
            "Create Docker compose for local development",
            "Set up monitoring with {tool}",
            "Configure auto-scaling for {tier} tier",
            "Create backup strategy for {database}",
            "Set up log aggregation with {stack}",
            "Configure SSL certificates for {domain}",
            "Implement blue-green deployment for {service}",
            "Set up secrets management with {tool}",
        ],
        "fills": {
            "project": ["main app", "microservices", "data pipeline"],
            "service": ["api", "worker", "scheduler", "gateway"],
            "tool": ["Prometheus", "Grafana", "Datadog", "New Relic"],
            "tier": ["web", "api", "worker"],
            "database": ["PostgreSQL", "MongoDB", "Redis"],
            "stack": ["ELK", "Loki", "CloudWatch"],
            "domain": ["api.example.com", "app.example.com"],
        },
    },
    "data": {
        "intents": [
            "Build ETL pipeline for {source} data",
            "Create data model for {domain}",
            "Optimize slow {query} query",
            "Set up data warehouse tables for {area}",
            "Implement data validation for {dataset}",
            "Create dashboard for {metrics}",
            "Build ML feature pipeline for {model}",
            "Set up data quality monitoring",
            "Migrate data from {source} to {target}",
            "Create data documentation for {dataset}",
        ],
        "fills": {
            "source": ["Salesforce", "Google Analytics", "internal API", "S3"],
            "domain": ["customer journey", "sales funnel", "product usage"],
            "query": ["customer analytics", "order history", "inventory"],
            "area": ["marketing", "sales", "product", "finance"],
            "dataset": ["user events", "transactions", "logs"],
            "metrics": ["revenue", "engagement", "conversion"],
            "model": ["churn prediction", "recommendation", "fraud detection"],
            "target": ["Snowflake", "BigQuery", "Redshift"],
        },
    },
    "mobile": {
        "intents": [
            "Build {screen} screen for iOS app",
            "Implement push notifications for {feature}",
            "Add offline support for {module}",
            "Fix crash in {component} on Android",
            "Optimize app startup time",
            "Implement deep linking for {feature}",
            "Add biometric authentication",
            "Create widget for {feature}",
            "Implement in-app purchases for {product}",
            "Add analytics tracking for {event}",
        ],
        "fills": {
            "screen": ["Home", "Profile", "Settings", "Detail", "Search"],
            "feature": ["messages", "orders", "updates", "reminders"],
            "module": ["cart", "favorites", "history"],
            "component": ["image loader", "list view", "navigation"],
            "product": ["premium", "subscription", "credits"],
            "event": ["purchase", "signup", "share"],
        },
    },
    "research": {
        "intents": [
            "Analyze papers on {topic}",
            "Compare approaches for {problem}",
            "Summarize findings on {subject}",
            "Find datasets for {domain}",
            "Review the methodology in {paper}",
            "Explain the key ideas in {technique}",
            "Identify gaps in {field} research",
            "Synthesize literature on {topic}",
            "Evaluate claims about {subject}",
            "Design experiment for {hypothesis}",
        ],
        "fills": {
            "topic": ["transformer attention", "SSM architectures", "memory networks", "uncertainty quantification"],
            "problem": ["long-context modeling", "few-shot learning", "continual learning"],
            "subject": ["scaling laws", "emergent abilities", "in-context learning"],
            "domain": ["NLP", "vision", "robotics", "healthcare"],
            "paper": ["recent SOTA", "baseline", "ablation study"],
            "technique": ["attention mechanism", "state space models", "mixture of experts"],
            "field": ["cognitive architectures", "neurosymbolic AI", "embodied AI"],
            "hypothesis": ["model scaling", "data quality", "architectural choices"],
        },
    },
    "writing": {
        "intents": [
            "Write documentation for {component}",
            "Create user guide for {feature}",
            "Draft blog post about {topic}",
            "Write technical spec for {project}",
            "Create README for {repo}",
            "Write release notes for {version}",
            "Draft email to {audience} about {topic}",
            "Create presentation on {subject}",
            "Write API documentation for {endpoint}",
            "Create onboarding guide for {role}",
        ],
        "fills": {
            "component": ["auth system", "payment flow", "search", "API"],
            "feature": ["new dashboard", "collaboration", "exports"],
            "topic": ["architecture decisions", "performance optimization", "security"],
            "project": ["redesign", "migration", "integration"],
            "repo": ["main app", "SDK", "CLI tool"],
            "version": ["v2.0", "Q1 release", "hotfix"],
            "audience": ["customers", "team", "stakeholders", "investors"],
            "subject": ["quarterly progress", "technical approach", "roadmap"],
            "endpoint": ["REST API", "GraphQL", "webhooks"],
            "role": ["new developers", "designers", "PMs"],
        },
    },
    "planning": {
        "intents": [
            "Create project plan for {project}",
            "Break down {epic} into tasks",
            "Estimate effort for {feature}",
            "Prioritize {backlog} backlog",
            "Create roadmap for {quarter}",
            "Plan sprint for {team}",
            "Define milestones for {goal}",
            "Identify dependencies for {project}",
            "Create timeline for {release}",
            "Plan resource allocation for {period}",
        ],
        "fills": {
            "project": ["platform migration", "new feature launch", "redesign"],
            "epic": ["user onboarding", "payment system", "search upgrade"],
            "feature": ["real-time sync", "export", "notifications"],
            "backlog": ["product", "tech debt", "bugs"],
            "quarter": ["Q1", "Q2", "H1"],
            "team": ["backend", "frontend", "platform"],
            "goal": ["launch MVP", "scale system", "improve UX"],
            "release": ["v2.0", "beta", "public launch"],
            "period": ["next month", "quarter", "year"],
        },
    },
    "debugging": {
        "intents": [
            "Debug the {symptom} issue in {component}",
            "Investigate why {feature} is slow",
            "Fix the memory leak in {service}",
            "Troubleshoot {error} errors in production",
            "Find the root cause of {symptom}",
            "Debug intermittent {issue} in {area}",
            "Investigate performance regression in {feature}",
            "Fix data inconsistency in {module}",
            "Debug authentication failures for {user_type}",
            "Troubleshoot deployment failure for {service}",
        ],
        "fills": {
            "symptom": ["timeout", "crash", "hanging", "data loss", "incorrect results"],
            "component": ["login", "checkout", "search", "sync"],
            "feature": ["dashboard load", "file upload", "search"],
            "service": ["API", "worker", "scheduler"],
            "error": ["500", "timeout", "connection refused", "OOM"],
            "issue": ["failure", "error", "timeout"],
            "area": ["production", "staging", "CI"],
            "module": ["billing", "inventory", "user accounts"],
            "user_type": ["admin", "regular user", "API client"],
        },
    },
    "design": {
        "intents": [
            "Design system architecture for {project}",
            "Create schema for {data_type}",
            "Design API contract for {service}",
            "Plan database structure for {feature}",
            "Design caching strategy for {use_case}",
            "Create event-driven architecture for {domain}",
            "Design error handling strategy",
            "Plan migration path from {old} to {new}",
            "Design testing strategy for {area}",
            "Create security model for {system}",
        ],
        "fills": {
            "project": ["microservices migration", "new product", "platform redesign"],
            "data_type": ["user events", "transactions", "configuration"],
            "service": ["payment", "notification", "search"],
            "feature": ["multi-tenancy", "versioning", "audit logs"],
            "use_case": ["session data", "API responses", "computed values"],
            "domain": ["order processing", "real-time updates", "notifications"],
            "old": ["monolith", "legacy API", "SQL"],
            "new": ["microservices", "REST", "NoSQL"],
            "area": ["API", "integration", "end-to-end"],
            "system": ["API", "admin panel", "data pipeline"],
        },
    },
}

# Natural language variations for more realism
INTENT_PREFIXES = [
    "", "I need to ", "Let's ", "Can you help me ", "We should ", 
    "I want to ", "Please ", "Help me ", "I'd like to ",
    "We need to ", "Time to ", "Got to ", "Need help with ",
]

INTENT_SUFFIXES = [
    "", " asap", " when you get a chance", " today", 
    " before the deadline", " for the demo", " urgently",
    " - it's blocking", " for the release",
]

# Ambiguous intents for adversarial scenarios
AMBIGUOUS_INTENTS = [
    "Do the thing we discussed",
    "Fix it",
    "Make it better",
    "You know what I mean",
    "Same as last time but different",
    "The usual",
    "What we talked about",
    "That feature thing",
    "The bug",
    "Make it work",
    "Can you look at it?",
    "The issue",
    "You know",
    "The thing",
    "Handle it",
]

# Error messages for adversarial scenarios
ERROR_MESSAGES = [
    "Connection timeout after 30s",
    "Permission denied: insufficient privileges",
    "Database connection failed",
    "Out of memory",
    "Rate limited: try again in 60s",
    "Invalid configuration",
    "Dependency not found",
    "SSL certificate expired",
    "Disk full",
    "Service unavailable",
    "Authentication failed",
    "Invalid API key",
    "Resource not found",
    "Conflict: resource already exists",
    "Validation failed: missing required field",
]


def generate_mt_hash() -> str:
    """Generate a random MT hash."""
    return hashlib.sha256(str(random.random()).encode()).hexdigest()[:32]


def fill_template(template: str, fills: Dict[str, List[str]]) -> str:
    """Fill a template with random values."""
    result = template
    for key, values in fills.items():
        placeholder = f"{{{key}}}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def naturalize_intent(intent: str) -> str:
    """Add natural language variation to intent."""
    prefix = random.choice(INTENT_PREFIXES) if random.random() > 0.4 else ""
    suffix = random.choice(INTENT_SUFFIXES) if random.random() > 0.7 else ""
    return prefix + intent + suffix


class EnhancedScenarioGenerator:
    """Generates diverse, realistic training scenarios for ODIN."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.scenario_count = {"planning": 0, "revision": 0, "adversarial": 0}
        self.all_domains = list(DOMAINS.keys())
    
    def generate_planning_scenario(self, difficulty: float = 0.3) -> Scenario:
        """Generate a planning scenario with clear goal and execution."""
        domain = random.choice(self.all_domains)
        domain_data = DOMAINS[domain]
        template = random.choice(domain_data["intents"])
        intent = naturalize_intent(fill_template(template, domain_data["fills"]))
        
        # Variable length based on difficulty
        num_steps = random.randint(2, 4) if difficulty < 0.5 else random.randint(3, 6)
        steps = []
        mt_hash = generate_mt_hash()
        
        # Step 0: Initial planning
        initial_confidence = 0.6 + random.random() * 0.2
        uncertainty = random.sample(["scope", "approach", "timeline", "dependencies"], 
                                   k=random.randint(0, 2))
        
        steps.append(ScenarioStep(
            t=0,
            input_state=InputState(
                intent=intent,
                mt_hash=mt_hash,
                mt_delta=[],
                feedback=None,
                system={"urgency": random.random() * 0.5 + 0.2, "timestamp": 1704825600},
            ),
            expected_output=ExpectedOutput(
                stance=Stance.PLAN,
                confidence=initial_confidence,
                action=Action.DECOMPOSE_TASK if len(intent) > 30 else Action.PLAN,
                uncertainty=uncertainty,
            ),
            reward=1.0,
        ))
        
        # Middle steps: Execution
        for t in range(1, num_steps - 1):
            confidence = min(0.95, initial_confidence + t * 0.05)
            action = random.choice([Action.EXECUTE, Action.DELEGATE, Action.WRITE_MT])
            
            steps.append(ScenarioStep(
                t=t,
                input_state=InputState(
                    intent=f"Proceeding with step {t}",
                    mt_hash=mt_hash,
                    mt_delta=[{"type": "progress", "step": t}],
                    feedback={"success": True, "step": t-1},
                    system={"urgency": 0.3 + random.random() * 0.2, "timestamp": 1704825600 + t * 300},
                ),
                expected_output=ExpectedOutput(
                    stance=Stance.EXECUTE,
                    confidence=confidence,
                    action=action,
                    uncertainty=[],
                ),
                reward=1.0,
            ))
        
        # Final step: Commit
        steps.append(ScenarioStep(
            t=num_steps - 1,
            input_state=InputState(
                intent="Complete and commit",
                mt_hash=mt_hash,
                mt_delta=[{"type": "complete"}],
                feedback={"success": True},
                system={"urgency": 0.2, "timestamp": 1704825600 + num_steps * 300},
            ),
            expected_output=ExpectedOutput(
                stance=Stance.EXECUTE,
                confidence=0.9 + random.random() * 0.08,
                action=Action.COMMIT,
                uncertainty=[],
            ),
            reward=1.0,
        ))
        
        self.scenario_count["planning"] += 1
        
        return Scenario(
            id=f"planning_{self.scenario_count['planning']:05d}",
            type=ScenarioType.PLANNING,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="simple" if num_steps <= 3 else "medium",
                required_pivots=0,
            ),
        )
    
    def generate_revision_scenario(self, difficulty: float = 0.5) -> Scenario:
        """Generate a scenario with goal revision mid-execution."""
        domain = random.choice(self.all_domains)
        domain_data = DOMAINS[domain]
        
        # Two different intents for the revision
        template1 = random.choice(domain_data["intents"])
        template2 = random.choice([t for t in domain_data["intents"] if t != template1])
        
        initial_intent = naturalize_intent(fill_template(template1, domain_data["fills"]))
        revised_intent = naturalize_intent(fill_template(template2, domain_data["fills"]))
        
        num_steps = random.randint(4, 8)
        pivot_step = random.randint(1, min(3, num_steps - 2))
        steps = []
        mt_hash = generate_mt_hash()
        
        for t in range(num_steps):
            if t == 0:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=initial_intent,
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.4 + random.random() * 0.2, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.7 + random.random() * 0.1,
                        action=Action.PLAN,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == pivot_step:
                # Revision point
                revision_intros = [
                    f"Actually, change of plans: {revised_intent}",
                    f"Wait, let's pivot to: {revised_intent}",
                    f"New priority: {revised_intent}",
                    f"Scratch that, we need to {revised_intent.lower()}",
                ]
                
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=random.choice(revision_intros),
                        mt_hash=generate_mt_hash(),
                        mt_delta=[{"type": "goal_revision", "old": initial_intent[:50]}],
                        feedback=None,
                        system={"urgency": 0.6 + random.random() * 0.2, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.REFLECT,
                        confidence=0.4 + random.random() * 0.2,
                        action=Action.BACKTRACK,
                        uncertainty=["new_scope", "prior_work_applicability"],
                    ),
                    reward=1.0,
                ))
            elif t == pivot_step + 1:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=revised_intent,
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.6 + random.random() * 0.1,
                        action=Action.DECOMPOSE_TASK,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps - 1:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Finalize",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "complete"}],
                        feedback={"success": True},
                        system={"urgency": 0.3, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.85 + random.random() * 0.1,
                        action=Action.COMMIT,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=f"Continue step {t}",
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback={"success": True},
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.7 + random.random() * 0.15,
                        action=Action.EXECUTE,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.scenario_count["revision"] += 1
        
        return Scenario(
            id=f"revision_{self.scenario_count['revision']:05d}",
            type=ScenarioType.REVISION,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="medium" if num_steps <= 5 else "complex",
                required_pivots=1,
            ),
        )
    
    def generate_adversarial_scenario(self, difficulty: float = 0.8) -> Scenario:
        """Generate adversarial scenario with ambiguity and errors."""
        domain = random.choice(self.all_domains)
        domain_data = DOMAINS[domain]
        
        num_steps = random.randint(5, 10)
        error_step = random.randint(2, num_steps - 3) if random.random() > 0.3 else -1
        clarification_step = 1
        steps = []
        mt_hash = generate_mt_hash()
        
        for t in range(num_steps):
            if t == 0:
                # Ambiguous initial intent
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=random.choice(AMBIGUOUS_INTENTS),
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.6 + random.random() * 0.3, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXPLORE,
                        confidence=0.2 + random.random() * 0.2,
                        action=Action.QUERY_USER,
                        uncertainty=["intent_unclear", "context_missing", "scope_unknown"],
                    ),
                    reward=1.0,
                ))
            elif t == clarification_step:
                # User clarifies
                template = random.choice(domain_data["intents"])
                clear_intent = fill_template(template, domain_data["fills"])
                
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=clear_intent,
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.7 + random.random() * 0.15,
                        action=Action.PLAN,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == error_step:
                # Error occurs
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Continue",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "error", "code": "EXEC_FAILED"}],
                        feedback={"success": False, "error": random.choice(ERROR_MESSAGES)},
                        system={"urgency": 0.8 + random.random() * 0.1, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.REFLECT,
                        confidence=0.3 + random.random() * 0.2,
                        action=Action.BACKTRACK,
                        uncertainty=["error_cause", "recovery_strategy"],
                    ),
                    reward=1.0,
                ))
            elif t == error_step + 1 and error_step > 0:
                # Recovery
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Try alternative approach",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "recovery"}],
                        feedback=None,
                        system={"urgency": 0.6, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.5 + random.random() * 0.15,
                        action=Action.PLAN,
                        uncertainty=["alternative_viability"],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps - 1:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Complete",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "complete"}],
                        feedback={"success": True},
                        system={"urgency": 0.2, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.8 + random.random() * 0.1,
                        action=Action.COMMIT,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=f"Step {t}",
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback={"success": True},
                        system={"urgency": 0.4 + random.random() * 0.1, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.7 + random.random() * 0.1,
                        action=Action.EXECUTE,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.scenario_count["adversarial"] += 1
        
        return Scenario(
            id=f"adversarial_{self.scenario_count['adversarial']:05d}",
            type=ScenarioType.ADVERSARIAL,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="complex",
                required_pivots=0,
                error_recovery=error_step > 0,
                ambiguity_level=0.7 + random.random() * 0.2,
            ),
        )
    
    def generate_scenario(self, difficulty: float) -> Scenario:
        """Generate a scenario based on difficulty-weighted distribution."""
        if difficulty < 0.3:
            probs = (0.8, 0.1, 0.1)
        elif difficulty < 0.7:
            probs = (0.5, 0.3, 0.2)
        else:
            probs = (0.3, 0.35, 0.35)
        
        scenario_type = random.choices(
            ["planning", "revision", "adversarial"],
            weights=probs,
            k=1
        )[0]
        
        if scenario_type == "planning":
            return self.generate_planning_scenario(difficulty)
        elif scenario_type == "revision":
            return self.generate_revision_scenario(difficulty)
        else:
            return self.generate_adversarial_scenario(difficulty)


def generate_dataset(
    output_dir: str,
    num_scenarios: int = 10000,
    seed: int = 42,
):
    """Generate a full training dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "planning").mkdir(exist_ok=True)
    (output_path / "revision").mkdir(exist_ok=True)
    (output_path / "adversarial").mkdir(exist_ok=True)
    
    generator = EnhancedScenarioGenerator(seed=seed)
    
    print(f"Generating {num_scenarios} enhanced scenarios...")
    print(f"Domains: {len(DOMAINS)} ({', '.join(DOMAINS.keys())})")
    
    for i in range(num_scenarios):
        difficulty = min(1.0, i / (num_scenarios * 0.6))
        scenario = generator.generate_scenario(difficulty)
        
        subdir = scenario.type.value
        filename = f"{scenario.id}.json"
        scenario.save(str(output_path / subdir / filename))
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_scenarios} scenarios")
    
    stats = {
        "total": num_scenarios,
        "planning": generator.scenario_count["planning"],
        "revision": generator.scenario_count["revision"],
        "adversarial": generator.scenario_count["adversarial"],
        "domains": list(DOMAINS.keys()),
    }
    
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset generation complete!")
    print(f"  Planning: {stats['planning']}")
    print(f"  Revision: {stats['revision']}")
    print(f"  Adversarial: {stats['adversarial']}")
    print(f"  Domains: {len(stats['domains'])}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ODIN training scenarios")
    parser.add_argument("--output", type=str, default="../scenarios",
                       help="Output directory")
    parser.add_argument("--num", type=int, default=10000,
                       help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    generate_dataset(args.output, args.num, args.seed)
