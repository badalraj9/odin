"""
ODIN Scenario Generator V2 - Elite Tier
----------------------------------------
Competition-grade training data for cognitive reasoning.
"""

import random
import hashlib
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from schema import (
    Scenario, ScenarioStep, InputState, ExpectedOutput, ScenarioMetadata,
    ScenarioType, Stance, Action
)

# =============================================================================
# EXPANDED DOMAINS (15 domains)
# =============================================================================

DOMAINS = {
    "backend": {
        "intents": [
            "Build a REST API for {entity} management with {auth} authentication",
            "Create CRUD endpoints for {entity} with {validation} validation",
            "Add {cache} caching to the {service} service",
            "Fix the {issue} issue in the {feature} handler causing {symptom}",
            "Implement rate limiting for the {endpoint} endpoint with {limit} requests per minute",
            "Add database migrations for the {table} table with {constraint} constraints",
            "Set up {queue} queue for background {job} processing",
            "Create webhook handler for {integration} with retry logic",
            "Implement pagination for the {entity} list with {sort} sorting",
            "Add comprehensive error handling to the {service} service",
            "Refactor {component} to use {pattern} pattern for better testability",
            "Optimize {query} query that's taking {time}ms on average",
        ],
        "fills": {
            "entity": ["user", "product", "order", "payment", "subscription", "invoice", "customer", "inventory", "booking", "review"],
            "service": ["auth", "billing", "notification", "analytics", "search", "recommendation", "shipping", "inventory"],
            "feature": ["login", "checkout", "search", "upload", "export", "sync", "webhook", "batch"],
            "endpoint": ["public", "admin", "webhook", "batch", "internal"],
            "table": ["users", "orders", "products", "transactions", "logs", "sessions", "events"],
            "integration": ["Stripe", "SendGrid", "Twilio", "Slack", "GitHub", "AWS", "Shopify"],
            "auth": ["JWT", "OAuth2", "API key", "session-based"],
            "validation": ["schema", "runtime", "async"],
            "cache": ["Redis", "in-memory", "distributed"],
            "issue": ["memory leak", "race condition", "deadlock", "N+1 query"],
            "symptom": ["slowness", "crashes", "data corruption", "timeouts"],
            "limit": ["100", "500", "1000"],
            "constraint": ["unique", "foreign key", "check"],
            "queue": ["Redis", "RabbitMQ", "SQS"],
            "job": ["email", "report", "sync", "cleanup"],
            "sort": ["timestamp", "relevance", "alphabetical"],
            "component": ["user service", "payment handler", "auth middleware"],
            "pattern": ["repository", "factory", "strategy"],
            "query": ["user analytics", "order history", "inventory check"],
            "time": ["2000", "5000", "10000"],
        },
    },
    "frontend": {
        "intents": [
            "Create a {component} component with {state} state management",
            "Build responsive {page} page with mobile-first design",
            "Add dark mode with system preference detection to {area}",
            "Implement lazy loading for {feature} with {placeholder} placeholder",
            "Fix CSS {bug} in {component} affecting {browser}",
            "Add form validation to {form} with real-time error display",
            "Create reusable {element} with {variants} variants",
            "Implement drag and drop for {feature} with accessibility support",
            "Add keyboard navigation to {component} following WCAG guidelines",
            "Optimize bundle size by code-splitting the {module} module",
            "Implement {animation} animation for {interaction} interactions",
            "Add offline support with service worker for {feature}",
        ],
        "fills": {
            "component": ["Dashboard", "Settings", "Profile", "Sidebar", "Modal", "DataTable", "Chart", "Wizard", "Timeline"],
            "page": ["landing", "pricing", "about", "contact", "dashboard", "onboarding", "checkout", "search results"],
            "area": ["header", "sidebar", "main content", "footer", "navigation"],
            "feature": ["images", "comments", "feed", "notifications", "media gallery"],
            "form": ["login", "registration", "checkout", "contact", "settings", "profile edit"],
            "element": ["Button", "Input", "Card", "Dropdown", "Toast", "Badge", "Avatar"],
            "module": ["admin", "public", "shared", "analytics"],
            "state": ["Redux", "Context", "Zustand", "local"],
            "placeholder": ["skeleton", "spinner", "blur"],
            "bug": ["overflow", "z-index", "flexbox", "grid alignment"],
            "browser": ["Safari", "Firefox", "mobile Chrome"],
            "variants": ["3", "5", "multiple"],
            "animation": ["fade", "slide", "scale", "spring"],
            "interaction": ["hover", "click", "scroll", "page transition"],
        },
    },
    "devops": {
        "intents": [
            "Set up CI/CD pipeline for {project} with {stages} stages",
            "Configure Kubernetes deployment with {replicas} replicas and auto-scaling",
            "Create Docker compose for local development with {services} services",
            "Set up monitoring with {tool} including custom {metric} metrics",
            "Configure auto-scaling based on {metric} with {threshold} threshold",
            "Create disaster recovery plan with {rpo} RPO and {rto} RTO",
            "Set up log aggregation with alerting for {pattern} patterns",
            "Configure SSL/TLS with auto-renewal for {domain}",
            "Implement blue-green deployment with automatic rollback",
            "Set up secrets rotation with {tool} every {interval}",
            "Configure network security with {firewall} rules",
            "Set up cost monitoring with alerts at {threshold} threshold",
        ],
        "fills": {
            "project": ["main app", "microservices", "data pipeline", "ML platform"],
            "service": ["api", "worker", "scheduler", "gateway", "database"],
            "tool": ["Prometheus", "Grafana", "Datadog", "New Relic", "CloudWatch"],
            "tier": ["web", "api", "worker", "cache"],
            "database": ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch"],
            "stack": ["ELK", "Loki", "CloudWatch", "Splunk"],
            "domain": ["api.example.com", "app.example.com", "*.example.com"],
            "stages": ["build, test, deploy", "lint, test, staging, prod"],
            "replicas": ["3", "5", "10"],
            "services": ["5", "8", "12"],
            "metric": ["CPU", "memory", "request latency", "error rate"],
            "threshold": ["70%", "80%", "90%"],
            "rpo": ["1 hour", "4 hours", "24 hours"],
            "rto": ["15 minutes", "1 hour", "4 hours"],
            "pattern": ["error spike", "latency increase", "memory leak"],
            "interval": ["30 days", "90 days", "monthly"],
            "firewall": ["ingress", "egress", "WAF"],
        },
    },
    "data": {
        "intents": [
            "Build ETL pipeline for {source} data with {frequency} refresh",
            "Create data model for {domain} with {normalization} normalization",
            "Optimize slow {query} query running on {volume} rows",
            "Set up data warehouse tables with {partitioning} partitioning",
            "Implement data validation with {strategy} strategy for {dataset}",
            "Create real-time dashboard for {metrics} with {refresh} refresh",
            "Build ML feature pipeline for {model} with feature store",
            "Set up data quality monitoring with {threshold}% threshold",
            "Migrate {volume} data from {source} to {target} with zero downtime",
            "Create data lineage documentation for {dataset}",
            "Implement CDC for {table} with {latency} latency requirement",
            "Build data anonymization pipeline for {regulation} compliance",
        ],
        "fills": {
            "source": ["Salesforce", "Google Analytics", "internal API", "S3", "Kafka", "PostgreSQL"],
            "domain": ["customer journey", "sales funnel", "product usage", "financial reporting"],
            "query": ["customer analytics", "order history", "inventory", "revenue rollup"],
            "area": ["marketing", "sales", "product", "finance", "operations"],
            "dataset": ["user events", "transactions", "logs", "sessions"],
            "metrics": ["revenue", "engagement", "conversion", "retention"],
            "model": ["churn prediction", "recommendation", "fraud detection", "demand forecasting"],
            "target": ["Snowflake", "BigQuery", "Redshift", "Databricks"],
            "frequency": ["hourly", "daily", "real-time"],
            "normalization": ["3NF", "star schema", "snowflake"],
            "volume": ["1M", "100M", "1B"],
            "partitioning": ["date", "hash", "range"],
            "strategy": ["schema", "statistical", "ML-based"],
            "refresh": ["1 second", "5 seconds", "1 minute"],
            "threshold": ["95", "99", "99.9"],
            "latency": ["sub-second", "minute", "5 minute"],
            "table": ["orders", "events", "users"],
            "regulation": ["GDPR", "CCPA", "HIPAA"],
        },
    },
    "mobile": {
        "intents": [
            "Build {screen} screen with {animation} transitions",
            "Implement push notifications for {feature} with {targeting} targeting",
            "Add offline mode for {module} with sync on reconnect",
            "Fix {crash} crash in {component} on {platform}",
            "Optimize app startup to under {time} seconds",
            "Implement deep linking for {feature} with universal links",
            "Add biometric authentication with {fallback} fallback",
            "Create home screen widget showing {data}",
            "Implement in-app purchases with {restore} restore functionality",
            "Add analytics tracking for {events} user events",
            "Reduce app size by {percentage}% with asset optimization",
            "Implement app clips / instant apps for {feature}",
        ],
        "fills": {
            "screen": ["Home", "Profile", "Settings", "Detail", "Search", "Onboarding", "Checkout"],
            "feature": ["messages", "orders", "updates", "reminders", "promotions"],
            "module": ["cart", "favorites", "history", "content"],
            "component": ["image loader", "list view", "navigation", "video player"],
            "product": ["premium", "subscription", "credits", "one-time"],
            "events": ["screen views", "purchases", "shares", "all"],
            "animation": ["slide", "fade", "shared element"],
            "targeting": ["segment", "personalized", "geo-based"],
            "crash": ["memory", "null pointer", "network", "layout"],
            "platform": ["iOS 14", "Android 10", "older devices"],
            "time": ["2", "3", "1.5"],
            "fallback": ["PIN", "password", "pattern"],
            "data": ["weather", "tasks", "notifications"],
            "restore": ["server-side", "receipt", "cross-platform"],
            "percentage": ["20", "30", "40"],
        },
    },
    "security": {
        "intents": [
            "Implement {auth} authentication with {mfa} MFA",
            "Add rate limiting to prevent {attack} attacks",
            "Set up security headers including {header}",
            "Implement input sanitization against {vulnerability}",
            "Add audit logging for {actions} with tamper protection",
            "Set up secrets management with {rotation} rotation",
            "Implement RBAC with {granularity} granularity",
            "Add encryption for {data} using {algorithm}",
            "Set up intrusion detection for {vector} attacks",
            "Implement session management with {timeout} timeout",
            "Add API security with {mechanism} validation",
            "Set up compliance reporting for {standard}",
        ],
        "fills": {
            "auth": ["OAuth2", "SAML", "OpenID Connect"],
            "mfa": ["TOTP", "SMS", "hardware key"],
            "attack": ["brute force", "DDoS", "credential stuffing"],
            "header": ["CSP", "HSTS", "X-Frame-Options"],
            "vulnerability": ["XSS", "SQL injection", "SSRF"],
            "actions": ["admin", "financial", "data access"],
            "rotation": ["automatic", "manual", "30-day"],
            "granularity": ["role", "permission", "resource"],
            "data": ["PII", "financial", "credentials"],
            "algorithm": ["AES-256", "RSA", "ChaCha20"],
            "vector": ["network", "application", "insider"],
            "timeout": ["15 min", "30 min", "1 hour"],
            "mechanism": ["JWT", "API key", "OAuth"],
            "standard": ["SOC2", "PCI-DSS", "ISO27001"],
        },
    },
    "ml": {
        "intents": [
            "Train {model} model on {dataset} with {metric} optimization",
            "Deploy model to {platform} with {latency} latency SLA",
            "Implement A/B testing for {feature} with {significance} significance",
            "Build feature store for {features} features",
            "Set up model monitoring for {drift} drift detection",
            "Create training pipeline with {experiment} experiment tracking",
            "Optimize inference to {latency} with {technique}",
            "Implement online learning for {use_case}",
            "Build explainability for {model} using {method}",
            "Set up automated retraining on {trigger}",
            "Create evaluation framework with {metrics} metrics",
            "Implement model versioning with {rollback} rollback",
        ],
        "fills": {
            "model": ["classification", "regression", "recommendation", "NLP", "vision"],
            "dataset": ["1M samples", "streaming", "imbalanced"],
            "metric": ["accuracy", "F1", "AUC", "RMSE"],
            "platform": ["Kubernetes", "SageMaker", "Vertex AI"],
            "latency": ["10ms", "50ms", "100ms"],
            "significance": ["95%", "99%"],
            "features": ["100", "500", "1000"],
            "drift": ["data", "concept", "prediction"],
            "experiment": ["MLflow", "W&B", "Neptune"],
            "technique": ["quantization", "pruning", "distillation"],
            "use_case": ["recommendations", "fraud", "personalization"],
            "method": ["SHAP", "LIME", "attention"],
            "trigger": ["schedule", "data change", "performance drop"],
            "metrics": ["business", "fairness", "latency"],
            "rollback": ["instant", "gradual", "shadow"],
        },
    },
    "testing": {
        "intents": [
            "Set up {type} testing framework with {coverage}% coverage target",
            "Create test fixtures for {component} with {data} data",
            "Implement load testing for {scenario} scenario",
            "Add visual regression testing for {pages} pages",
            "Create chaos engineering tests for {failure} failures",
            "Set up contract testing between {services} services",
            "Implement security testing with {tool}",
            "Add accessibility testing with {standard} compliance",
            "Create performance benchmarks for {operation}",
            "Set up E2E tests with {browser} browser automation",
            "Implement mutation testing with {threshold}% threshold",
            "Add fuzz testing for {input} inputs",
        ],
        "fills": {
            "type": ["unit", "integration", "E2E", "contract"],
            "coverage": ["80", "90", "95"],
            "component": ["API", "database", "auth", "payment"],
            "data": ["mock", "synthetic", "production-like"],
            "scenario": ["peak load", "sustained load", "spike"],
            "pages": ["critical path", "all", "checkout flow"],
            "failure": ["network", "database", "service"],
            "services": ["frontend-backend", "microservices", "external API"],
            "tool": ["OWASP ZAP", "Burp Suite", "Snyk"],
            "standard": ["WCAG 2.1 AA", "Section 508"],
            "operation": ["database query", "API response", "page load"],
            "browser": ["Playwright", "Cypress", "Selenium"],
            "threshold": ["60", "70", "80"],
            "input": ["API", "file upload", "form"],
        },
    },
    "architecture": {
        "intents": [
            "Design {style} architecture for {scale} scale",
            "Create migration plan from {current} to {target}",
            "Design event-driven system for {domain} with {guarantee} guarantees",
            "Plan multi-region deployment for {latency} latency",
            "Design caching strategy with {layers} layers",
            "Create API versioning strategy for {consumers} consumers",
            "Design fault-tolerant system with {availability} availability",
            "Plan database sharding for {growth} growth",
            "Design async processing for {throughput} throughput",
            "Create service mesh for {services} services",
            "Design observability stack for {complexity} complexity",
            "Plan capacity for {projection} traffic projection",
        ],
        "fills": {
            "style": ["microservices", "modular monolith", "serverless", "event-driven"],
            "scale": ["startup", "million users", "enterprise"],
            "current": ["monolith", "legacy", "on-prem"],
            "target": ["cloud-native", "microservices", "hybrid"],
            "domain": ["e-commerce", "real-time", "financial"],
            "guarantee": ["at-least-once", "exactly-once", "best-effort"],
            "latency": ["sub-100ms global", "regional", "geo-distributed"],
            "layers": ["2", "3", "distributed"],
            "consumers": ["internal", "public", "partner"],
            "availability": ["99.9%", "99.99%", "99.999%"],
            "growth": ["10x", "100x", "1000x"],
            "throughput": ["10K/s", "100K/s", "1M/s"],
            "services": ["10", "50", "100+"],
            "complexity": ["medium", "high", "critical"],
            "projection": ["3 month", "1 year", "3 year"],
        },
    },
    "research": {
        "intents": [
            "Analyze papers on {topic} from {venue}",
            "Compare {approaches} for {problem}",
            "Summarize findings on {subject} for {audience}",
            "Find datasets for {domain} with {size} samples",
            "Review methodology in {paper} for {goal}",
            "Identify gaps in {field} research",
            "Design experiment to test {hypothesis}",
            "Evaluate reproducibility of {claim}",
            "Synthesize literature on {topic}",
            "Create research roadmap for {goal}",
        ],
        "fills": {
            "topic": ["transformer architectures", "SSM", "efficient attention", "memory networks"],
            "venue": ["NeurIPS 2024", "ICML", "ACL", "arxiv"],
            "approaches": ["3", "5", "recent"],
            "problem": ["long-context", "efficiency", "generalization"],
            "subject": ["scaling laws", "emergent abilities", "in-context learning"],
            "audience": ["team", "executives", "paper"],
            "domain": ["NLP", "vision", "multimodal", "robotics"],
            "size": ["10K", "1M", "100M"],
            "paper": ["baseline", "SOTA", "ablation"],
            "goal": ["practical application", "publication", "product"],
            "field": ["cognitive AI", "neurosymbolic", "embodied"],
            "hypothesis": ["architecture matters", "data quality", "scale"],
            "claim": ["performance", "efficiency", "capability"],
        },
    },
}

# =============================================================================
# STEP TEMPLATES (Rich descriptions)
# =============================================================================

EXECUTION_STEPS = [
    "Implementing {component} logic as discussed",
    "Writing tests for {feature} functionality",
    "Connecting {service_a} to {service_b}",
    "Setting up {tool} configuration",
    "Adding validation for {input}",
    "Refactoring {module} for clarity",
    "Deploying changes to {environment}",
    "Running integration tests",
    "Updating documentation for {change}",
    "Reviewing code with team",
]

PLANNING_STEPS = [
    "Breaking down the {scope} scope into sub-tasks",
    "Estimating complexity of {component}",
    "Identifying dependencies with {team} team",
    "Creating technical design for {feature}",
    "Planning database schema changes",
    "Mapping out API contract",
    "Defining acceptance criteria",
    "Scheduling work across sprints",
]

EXPLORATION_STEPS = [
    "Researching existing solutions for {problem}",
    "Investigating root cause of {issue}",
    "Gathering requirements from {stakeholder}",
    "Analyzing trade-offs between {option_a} and {option_b}",
    "Exploring API documentation for {service}",
    "Testing hypothesis about {theory}",
    "Reviewing similar implementations",
    "Collecting metrics on current behavior",
]

REFLECTION_STEPS = [
    "Evaluating approach after discovering {finding}",
    "Reconsidering priority due to {change}",
    "Assessing impact of {blocker} on timeline",
    "Reviewing feedback from {source}",
    "Analyzing test failures in {area}",
    "Rethinking architecture after {insight}",
    "Adjusting plan based on {constraint}",
]

# =============================================================================
# SCENARIO PATTERNS
# =============================================================================

USER_CLARIFICATIONS = [
    "Oh I meant {clarification}",
    "Sorry, to clarify: {clarification}",
    "Actually, {clarification}",
    "Let me be more specific: {clarification}",
    "{clarification}",
]

INTERRUPTIONS = [
    "Wait, we need to handle {urgent} first - it's urgent",
    "Stop that, {blocker} is blocking production",
    "Can you pause and help with {urgent}?",
    "Priority change: {urgent} is now P0",
]

ERRORS = [
    "Error: {error}",
    "Failed: {error}",
    "{error}",
    "Unexpected issue: {error}",
]

ERROR_DETAILS = [
    "Connection timeout to {service}",
    "Permission denied for {resource}",
    "Rate limited, retry in {time}",
    "Invalid response from {api}",
    "Out of memory in {component}",
    "Deadlock detected in {module}",
    "Schema validation failed for {data}",
    "Dependency {package} not found",
    "SSL certificate expired for {domain}",
    "Database connection pool exhausted",
]


def fill_template(template: str, domain_fills: Dict = None) -> str:
    """Fill template with random values."""
    result = template
    fills = domain_fills or {}
    
    # Also fill generic placeholders
    generic_fills = {
        "component": ["user service", "payment handler", "auth module", "API gateway"],
        "service_a": ["backend", "frontend", "worker", "cache"],
        "service_b": ["database", "queue", "API", "storage"],
        "tool": ["Docker", "Terraform", "Jenkins", "GitHub Actions"],
        "input": ["user data", "request body", "file upload", "parameters"],
        "module": ["auth", "billing", "core", "utils"],
        "environment": ["staging", "production", "dev"],
        "change": ["new feature", "bug fix", "refactor"],
        "scope": ["entire", "limited", "phased"],
        "team": ["frontend", "platform", "data", "security"],
        "problem": ["performance", "reliability", "scalability"],
        "issue": ["timeout", "memory leak", "crash"],
        "stakeholder": ["product", "engineering", "customer"],
        "option_a": ["approach A", "solution 1", "current way"],
        "option_b": ["approach B", "solution 2", "new approach"],
        "service": ["external API", "internal service", "third party"],
        "theory": ["root cause", "hypothesis", "assumption"],
        "finding": ["new constraint", "unexpected behavior", "edge case"],
        "blocker": ["external dependency", "approval needed", "bug found"],
        "source": ["code review", "testing", "user feedback"],
        "area": ["critical path", "edge cases", "integration"],
        "insight": ["performance data", "user feedback", "new requirement"],
        "constraint": ["timeline", "resource", "technical"],
        "clarification": ["the specific requirement", "which version", "the exact scope"],
        "urgent": ["production issue", "security alert", "customer escalation"],
        "error": ["connection failed", "timeout", "validation error"],
        "time": ["60s", "5m", "1h"],
        "api": ["external API", "payment service", "auth service"],
        "resource": ["database", "file", "API endpoint"],
        "data": ["request", "response", "payload"],
        "package": ["dependency", "module", "library"],
        "domain": ["api.example.com", "service.internal"],
        "feature": ["new feature", "enhancement", "fix"],
    }
    
    all_fills = {**generic_fills, **fills}
    
    for key, values in all_fills.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    
    return result


def generate_mt_hash() -> str:
    return hashlib.sha256(str(random.random()).encode()).hexdigest()[:32]


class EliteScenarioGenerator:
    """Elite-tier scenario generator with balanced stance distribution."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.counts = {"planning": 0, "revision": 0, "adversarial": 0, 
                       "exploration": 0, "interruption": 0, "multi_step": 0}
        self.domains = list(DOMAINS.keys())
        self.stance_balance = {"EXPLORE": 0, "PLAN": 0, "EXECUTE": 0, "REFLECT": 0}
    
    def _pick_stance_balanced(self, preferred: Stance) -> Stance:
        """Pick stance with rebalancing toward underrepresented."""
        total = sum(self.stance_balance.values()) + 1
        ratios = {k: v/total for k, v in self.stance_balance.items()}
        
        # If preferred is overrepresented, consider switching
        target_ratio = 0.25
        if ratios.get(preferred.name, 0) > target_ratio + 0.1:
            # Pick underrepresented stance
            under = [k for k, v in ratios.items() if v < target_ratio - 0.05]
            if under:
                picked = Stance[random.choice(under)]
                self.stance_balance[picked.name] = self.stance_balance.get(picked.name, 0) + 1
                return picked
        
        self.stance_balance[preferred.name] = self.stance_balance.get(preferred.name, 0) + 1
        return preferred
    
    def generate_exploration_scenario(self, difficulty: float = 0.5) -> Scenario:
        """Scenario focused on EXPLORE stance - gathering information."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        
        # Vague initial intent
        vague_intents = [
            "I'm thinking about {concept}",
            "What's the best way to handle {concept}?",
            "Help me understand {concept}",
            "Not sure how to approach {concept}",
            "Need to figure out {concept}",
        ]
        
        concepts = list(domain_data["fills"].keys())[:5]
        concept = random.choice(domain_data["fills"][random.choice(concepts)])
        initial_intent = random.choice(vague_intents).replace("{concept}", concept)
        
        num_steps = random.randint(4, 7)
        steps = []
        mt_hash = generate_mt_hash()
        
        for t in range(num_steps):
            if t == 0:
                # Start exploring
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=initial_intent,
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.3 + random.random() * 0.2, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXPLORE,
                        confidence=0.3 + random.random() * 0.2,
                        action=Action.QUERY_USER,
                        uncertainty=["scope", "requirements", "constraints"],
                    ),
                    reward=1.0,
                ))
            elif t < num_steps // 2:
                # Information gathering
                step_desc = fill_template(random.choice(EXPLORATION_STEPS), domain_data["fills"])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=step_desc,
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "research", "step": t}],
                        feedback={"found": random.choice(["relevant info", "partial match", "needs more"])},
                        system={"urgency": 0.3, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXPLORE,
                        confidence=0.4 + t * 0.05,
                        action=random.choice([Action.EXPLORE, Action.QUERY_USER, Action.WRITE_MT]),
                        uncertainty=["completeness"],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps // 2:
                # Transition to planning
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="I think I understand now, let's plan",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "understanding_complete"}],
                        feedback=None,
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.7 + random.random() * 0.1,
                        action=Action.PLAN,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                # Execute
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill_template(random.choice(EXECUTION_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback={"success": True},
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.8 + random.random() * 0.1,
                        action=Action.EXECUTE,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["exploration"] += 1
        
        return Scenario(
            id=f"exploration_{self.counts['exploration']:05d}",
            type=ScenarioType.PLANNING,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="exploration",
                required_pivots=0,
            ),
        )
    
    def generate_interruption_scenario(self, difficulty: float = 0.7) -> Scenario:
        """Scenario with priority interruption mid-execution."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        
        template = random.choice(domain_data["intents"])
        initial_intent = fill_template(template, domain_data["fills"])
        
        num_steps = random.randint(6, 10)
        interrupt_step = random.randint(2, num_steps - 4)
        steps = []
        mt_hash = generate_mt_hash()
        in_interruption = False
        
        for t in range(num_steps):
            if t == 0:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=initial_intent,
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.75 + random.random() * 0.1,
                        action=Action.DECOMPOSE_TASK,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == interrupt_step:
                # INTERRUPTION
                interrupt_text = fill_template(random.choice(INTERRUPTIONS))
                in_interruption = True
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=interrupt_text,
                        mt_hash=generate_mt_hash(),
                        mt_delta=[{"type": "priority_change"}],
                        feedback=None,
                        system={"urgency": 0.95, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.REFLECT,
                        confidence=0.4 + random.random() * 0.2,
                        action=Action.BACKTRACK,
                        uncertainty=["context_switch", "original_task_state"],
                    ),
                    reward=1.0,
                ))
            elif in_interruption and t <= interrupt_step + 2:
                # Handle interruption
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Handling urgent issue",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "urgent_handling"}],
                        feedback={"urgent_status": "in_progress" if t == interrupt_step + 1 else "resolved"},
                        system={"urgency": 0.8 if t == interrupt_step + 1 else 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.7 + random.random() * 0.1,
                        action=Action.EXECUTE if t == interrupt_step + 1 else Action.COMMIT,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
                if t == interrupt_step + 2:
                    in_interruption = False
            elif t == interrupt_step + 3:
                # Resume original task
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Resuming original task",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "resume"}],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.REFLECT,
                        confidence=0.6 + random.random() * 0.1,
                        action=Action.PLAN,
                        uncertainty=["state_recovery"],
                    ),
                    reward=1.0,
                ))
            else:
                # Normal execution
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill_template(random.choice(EXECUTION_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback={"success": True},
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.75 + random.random() * 0.1,
                        action=Action.EXECUTE if t < num_steps - 1 else Action.COMMIT,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["interruption"] += 1
        
        return Scenario(
            id=f"interruption_{self.counts['interruption']:05d}",
            type=ScenarioType.REVISION,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="interruption",
                required_pivots=1,
            ),
        )
    
    def generate_multi_step_reasoning(self, difficulty: float = 0.6) -> Scenario:
        """Multi-step task with dependencies and reflection points."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        
        # Complex multi-part task
        template = random.choice(domain_data["intents"])
        task = fill_template(template, domain_data["fills"])
        
        phases = ["planning", "exploration", "implementation", "testing", "completion"]
        num_steps = random.randint(8, 12)
        steps = []
        mt_hash = generate_mt_hash()
        
        for t in range(num_steps):
            phase_idx = min(t * len(phases) // num_steps, len(phases) - 1)
            phase = phases[phase_idx]
            
            if phase == "planning":
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=task if t == 0 else fill_template(random.choice(PLANNING_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[] if t == 0 else [{"type": "plan_step"}],
                        feedback=None if t == 0 else {"phase": "planning"},
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.6 + t * 0.02,
                        action=Action.DECOMPOSE_TASK if t == 0 else Action.PLAN,
                        uncertainty=["scope"] if t == 0 else [],
                    ),
                    reward=1.0,
                ))
            elif phase == "exploration":
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill_template(random.choice(EXPLORATION_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "explore"}],
                        feedback={"phase": "exploration"},
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXPLORE,
                        confidence=0.5 + random.random() * 0.2,
                        action=Action.EXPLORE,
                        uncertainty=["approach"],
                    ),
                    reward=1.0,
                ))
            elif phase == "implementation":
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill_template(random.choice(EXECUTION_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "impl"}],
                        feedback={"success": True},
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.7 + random.random() * 0.15,
                        action=Action.EXECUTE,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif phase == "testing":
                test_passed = random.random() > 0.3
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Running tests and validation",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "test", "passed": test_passed}],
                        feedback={"success": test_passed, "tests": "passed" if test_passed else "1 failed"},
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE if test_passed else Stance.REFLECT,
                        confidence=0.8 if test_passed else 0.5,
                        action=Action.EXECUTE if test_passed else Action.BACKTRACK,
                        uncertainty=[] if test_passed else ["test_failure"],
                    ),
                    reward=1.0,
                ))
            else:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Finalizing and completing",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "complete"}],
                        feedback={"success": True},
                        system={"urgency": 0.3, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.9 + random.random() * 0.08,
                        action=Action.COMMIT,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["multi_step"] += 1
        
        return Scenario(
            id=f"multistep_{self.counts['multi_step']:05d}",
            type=ScenarioType.PLANNING,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="multi_step",
                required_pivots=0,
            ),
        )
    
    def generate_adversarial_v2(self, difficulty: float = 0.8) -> Scenario:
        """Hardened adversarial with errors, ambiguity, and edge cases."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        
        num_steps = random.randint(7, 12)
        ambiguity_step = 0
        error_step = random.randint(3, num_steps - 3)
        steps = []
        mt_hash = generate_mt_hash()
        
        for t in range(num_steps):
            if t == ambiguity_step:
                # Ambiguous start
                ambiguous_intents = [
                    "Do the thing", "Fix it", "Make it better", "You know what I mean",
                    "Same as before but different", "Handle it", "The usual but updated",
                    "That issue we discussed", "The feature", "Make it work",
                ]
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=random.choice(ambiguous_intents),
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.7 + random.random() * 0.2, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXPLORE,
                        confidence=0.15 + random.random() * 0.15,
                        action=Action.QUERY_USER,
                        uncertainty=["intent", "scope", "context", "priority"],
                    ),
                    reward=1.0,
                ))
            elif t == 1:
                # User clarifies
                template = random.choice(domain_data["intents"])
                clear = fill_template(template, domain_data["fills"])
                clarification = random.choice(USER_CLARIFICATIONS).replace("{clarification}", clear)
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=clarification,
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "clarification"}],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.7 + random.random() * 0.1,
                        action=Action.PLAN,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == error_step:
                # Error occurs
                error = fill_template(random.choice(ERROR_DETAILS))
                error_msg = random.choice(ERRORS).replace("{error}", error)
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Continue",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "error"}],
                        feedback={"success": False, "error": error_msg},
                        system={"urgency": 0.85, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.REFLECT,
                        confidence=0.3 + random.random() * 0.2,
                        action=Action.BACKTRACK,
                        uncertainty=["error_cause", "recovery_path"],
                    ),
                    reward=1.0,
                ))
            elif t == error_step + 1:
                # Recovery attempt
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Trying alternative approach",
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
                # Complete
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Done",
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "complete"}],
                        feedback={"success": True},
                        system={"urgency": 0.2, "timestamp": 1704825600 + t * 300},
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
                # Normal execution
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill_template(random.choice(EXECUTION_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback={"success": True},
                        system={"urgency": 0.4 + random.random() * 0.1, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.7 + random.random() * 0.15,
                        action=Action.EXECUTE,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["adversarial"] += 1
        
        return Scenario(
            id=f"adversarial_{self.counts['adversarial']:05d}",
            type=ScenarioType.ADVERSARIAL,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="adversarial",
                required_pivots=1,
                error_recovery=True,
                ambiguity_level=0.8,
            ),
        )
    
    def generate_scenario(self, difficulty: float) -> Scenario:
        """Generate balanced scenario based on difficulty."""
        # Scenario type distribution
        if difficulty < 0.3:
            weights = [0.35, 0.35, 0.15, 0.1, 0.05]
        elif difficulty < 0.6:
            weights = [0.2, 0.25, 0.25, 0.15, 0.15]
        else:
            weights = [0.15, 0.15, 0.25, 0.2, 0.25]
        
        types = ["exploration", "multi_step", "planning", "interruption", "adversarial"]
        chosen = random.choices(types, weights=weights, k=1)[0]
        
        if chosen == "exploration":
            return self.generate_exploration_scenario(difficulty)
        elif chosen == "multi_step":
            return self.generate_multi_step_reasoning(difficulty)
        elif chosen == "planning":
            return self._generate_planning_scenario(difficulty)
        elif chosen == "interruption":
            return self.generate_interruption_scenario(difficulty)
        else:
            return self.generate_adversarial_v2(difficulty)
    
    def _generate_planning_scenario(self, difficulty: float = 0.4) -> Scenario:
        """Standard planning scenario (simplified)."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        template = random.choice(domain_data["intents"])
        intent = fill_template(template, domain_data["fills"])
        
        num_steps = random.randint(3, 6)
        steps = []
        mt_hash = generate_mt_hash()
        
        for t in range(num_steps):
            if t == 0:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=intent,
                        mt_hash=mt_hash,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.4 + random.random() * 0.2, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.PLAN,
                        confidence=0.7 + random.random() * 0.15,
                        action=Action.DECOMPOSE_TASK if len(intent) > 40 else Action.PLAN,
                        uncertainty=["scope"] if difficulty > 0.5 else [],
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
                        confidence=0.9 + random.random() * 0.08,
                        action=Action.COMMIT,
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill_template(random.choice(EXECUTION_STEPS)),
                        mt_hash=mt_hash,
                        mt_delta=[{"type": "progress"}],
                        feedback={"success": True},
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=Stance.EXECUTE,
                        confidence=0.75 + random.random() * 0.1,
                        action=random.choice([Action.EXECUTE, Action.DELEGATE]),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["planning"] += 1
        
        return Scenario(
            id=f"planning_{self.counts['planning']:05d}",
            type=ScenarioType.PLANNING,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain,
                complexity="standard",
                required_pivots=0,
            ),
        )


def generate_elite_dataset(output_dir: str, num_scenarios: int = 10000, seed: int = 42):
    """Generate elite-tier balanced dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in ["planning", "revision", "adversarial"]:
        (output_path / subdir).mkdir(exist_ok=True)
    
    generator = EliteScenarioGenerator(seed=seed)
    
    print(f"Generating {num_scenarios} ELITE scenarios...")
    print(f"Domains: {len(DOMAINS)}")
    
    for i in range(num_scenarios):
        difficulty = min(1.0, i / (num_scenarios * 0.6))
        scenario = generator.generate_scenario(difficulty)
        
        subdir = scenario.type.value
        filename = f"{scenario.id}.json"
        scenario.save(str(output_path / subdir / filename))
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_scenarios}")
    
    stats = {
        "total": num_scenarios,
        "counts": generator.counts,
        "stance_balance": generator.stance_balance,
        "domains": len(DOMAINS),
    }
    
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nElite dataset complete!")
    print(f"  Counts: {generator.counts}")
    print(f"  Stance balance: {generator.stance_balance}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ELITE ODIN scenarios")
    parser.add_argument("--output", type=str, default="../scenarios", help="Output dir")
    parser.add_argument("--num", type=int, default=10000, help="Num scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    args = parser.parse_args()
    generate_elite_dataset(args.output, args.num, args.seed)
