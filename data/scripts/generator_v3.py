"""
ODIN Dataset Generator V3 - PRODUCTION GRADE
=============================================
Perfect balanced dataset for 30M production model.
Targets 100K scenarios with elite quality.
"""

import random
import hashlib
import json
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from schema import (
    Scenario, ScenarioStep, InputState, ExpectedOutput, ScenarioMetadata,
    ScenarioType, Stance, Action
)

# =============================================================================
# 20 PRODUCTION DOMAINS
# =============================================================================

DOMAINS = {
    "backend_api": {
        "intents": [
            "Build REST API for {entity} with {auth} and rate limiting",
            "Create GraphQL resolvers for {entity} queries and mutations",
            "Implement {pattern} pattern for {service} service",
            "Optimize {query} query taking {latency}ms to under 50ms",
            "Add comprehensive error handling to {endpoint} endpoint",
            "Implement retry logic with exponential backoff for {integration}",
            "Create batch processing endpoint for {operation}",
            "Add request validation using {schema} schema",
            "Implement idempotency for {operation} operations",
            "Create health check and readiness endpoints",
        ],
        "entities": ["user", "product", "order", "payment", "subscription", "invoice", "booking", "review", "catalog", "inventory"],
        "auth": ["JWT", "OAuth2", "API key", "session", "mTLS"],
        "pattern": ["repository", "CQRS", "saga", "circuit breaker", "bulkhead"],
        "service": ["auth", "billing", "notification", "search", "recommendation", "audit"],
        "query": ["user analytics", "order aggregation", "inventory lookup", "search ranking"],
        "latency": ["500", "1000", "2000", "5000"],
        "endpoint": ["/users", "/orders", "/payments", "/webhooks", "/batch"],
        "integration": ["Stripe", "SendGrid", "Twilio", "AWS SQS", "Kafka"],
        "operation": ["import", "export", "sync", "migration"],
        "schema": ["JSON Schema", "Zod", "Joi", "OpenAPI"],
    },
    "frontend_react": {
        "intents": [
            "Create {component} with {state} state and {animation} animations",
            "Build {page} page with responsive design and {loading} loading",
            "Implement {feature} with optimistic updates and error boundaries",
            "Add real-time {sync} sync using WebSocket",
            "Create accessible {element} following WCAG 2.1 AA",
            "Optimize {metric} by implementing {technique}",
            "Build form with multi-step wizard and validation",
            "Implement infinite scroll for {list} with virtualization",
            "Add keyboard navigation to {component}",
            "Create {chart} chart with interactive tooltips",
        ],
        "component": ["Dashboard", "DataTable", "Modal", "Sidebar", "Timeline", "Kanban", "Calendar"],
        "state": ["Redux", "Zustand", "React Query", "Jotai"],
        "animation": ["Framer Motion", "CSS", "Spring"],
        "page": ["landing", "dashboard", "settings", "onboarding", "checkout"],
        "loading": ["skeleton", "progressive", "optimistic"],
        "feature": ["drag-drop", "file upload", "rich text editor", "color picker"],
        "sync": ["presence", "cursor", "document", "notification"],
        "element": ["dropdown", "combobox", "date picker", "slider"],
        "metric": ["LCP", "FID", "CLS", "bundle size"],
        "technique": ["code splitting", "lazy loading", "memoization", "virtualization"],
        "list": ["feed", "comments", "products", "messages"],
        "chart": ["line", "bar", "pie", "scatter", "heatmap"],
    },
    "devops_infra": {
        "intents": [
            "Set up {pipeline} CI/CD with {stages} stages",
            "Configure Kubernetes with {replicas} replicas and {scaling}",
            "Create Terraform modules for {resource}",
            "Set up {monitoring} with custom {metric} alerting",
            "Implement {deployment} deployment strategy",
            "Configure {secret} secrets management",
            "Set up disaster recovery with {rpo} RPO",
            "Create {log} log aggregation pipeline",
            "Configure network policies for {isolation}",
            "Set up cost monitoring with {threshold} alerts",
        ],
        "pipeline": ["GitHub Actions", "GitLab CI", "Jenkins", "ArgoCD"],
        "stages": ["lint, test, build, deploy", "test, staging, canary, prod"],
        "replicas": ["3", "5", "10", "auto"],
        "scaling": ["HPA", "VPA", "KEDA", "custom metrics"],
        "resource": ["VPC", "EKS", "RDS", "Lambda", "API Gateway"],
        "monitoring": ["Prometheus + Grafana", "Datadog", "New Relic", "CloudWatch"],
        "metric": ["latency p99", "error rate", "saturation", "throughput"],
        "deployment": ["blue-green", "canary", "rolling", "A/B"],
        "secret": ["Vault", "AWS Secrets Manager", "sealed secrets"],
        "rpo": ["15 min", "1 hour", "4 hours"],
        "log": ["ELK", "Loki", "Splunk", "CloudWatch"],
        "isolation": ["namespace", "network policy", "service mesh"],
        "threshold": ["80%", "90%", "$1000/day"],
    },
    "data_engineering": {
        "intents": [
            "Build ETL pipeline from {source} to {dest} with {schedule}",
            "Create {model} data model with {normalization}",
            "Optimize {query} running on {volume} rows",
            "Implement data validation with {threshold}% accuracy",
            "Set up CDC for {table} with {latency} latency",
            "Create feature store for {ml_features}",
            "Build data quality monitoring dashboard",
            "Implement data lineage tracking",
            "Create incremental processing for {dataset}",
            "Set up data governance for {compliance}",
        ],
        "source": ["PostgreSQL", "MongoDB", "Kafka", "S3", "API", "Salesforce"],
        "dest": ["Snowflake", "BigQuery", "Redshift", "Delta Lake", "Iceberg"],
        "schedule": ["hourly", "daily", "real-time", "event-driven"],
        "model": ["star schema", "snowflake", "data vault", "OBT"],
        "normalization": ["3NF", "denormalized", "hybrid"],
        "query": ["customer 360", "revenue rollup", "funnel analysis"],
        "volume": ["1M", "100M", "1B", "10B"],
        "threshold": ["95", "99", "99.9"],
        "table": ["orders", "events", "transactions", "users"],
        "latency": ["sub-second", "minute", "5 minutes"],
        "ml_features": ["user embeddings", "time series", "aggregations"],
        "dataset": ["events", "logs", "transactions"],
        "compliance": ["GDPR", "CCPA", "HIPAA", "SOX"],
    },
    "ml_ops": {
        "intents": [
            "Deploy {model_type} model with {latency}ms latency SLA",
            "Implement A/B testing with {significance}% significance",
            "Set up model monitoring for {drift} drift",
            "Create training pipeline with {tracking} tracking",
            "Optimize inference using {technique}",
            "Build feature serving with {freshness} freshness",
            "Implement model explainability using {method}",
            "Set up automated retraining on {trigger}",
            "Create evaluation framework with {metrics}",
            "Build shadow deployment for {model}",
        ],
        "model_type": ["classification", "regression", "ranking", "embedding", "LLM"],
        "latency": ["10", "50", "100", "500"],
        "significance": ["95", "99"],
        "drift": ["data", "concept", "prediction", "feature"],
        "tracking": ["MLflow", "W&B", "Neptune", "Vertex AI"],
        "technique": ["quantization", "pruning", "distillation", "batching"],
        "freshness": ["real-time", "hourly", "daily"],
        "method": ["SHAP", "LIME", "attention", "counterfactual"],
        "trigger": ["schedule", "data drift", "performance drop"],
        "metrics": ["business KPIs", "fairness", "latency", "accuracy"],
        "model": ["recommendation", "fraud", "search", "personalization"],
    },
    "security": {
        "intents": [
            "Implement {auth} with {mfa} MFA support",
            "Add protection against {attack} attacks",
            "Configure {headers} security headers",
            "Implement {encryption} encryption for data at rest",
            "Set up WAF rules for {protection}",
            "Create RBAC with {granularity} granularity",
            "Implement audit logging with {retention} retention",
            "Add rate limiting against {abuse}",
            "Set up vulnerability scanning in {stage}",
            "Create incident response automation",
        ],
        "auth": ["OAuth2 + PKCE", "SAML", "OpenID Connect", "passkeys"],
        "mfa": ["TOTP", "WebAuthn", "push notification", "SMS"],
        "attack": ["SQL injection", "XSS", "CSRF", "SSRF", "RCE"],
        "headers": ["CSP", "HSTS", "X-Frame-Options", "all"],
        "encryption": ["AES-256-GCM", "ChaCha20", "field-level"],
        "protection": ["bot", "DDoS", "credential stuffing", "API abuse"],
        "granularity": ["role", "permission", "resource", "field"],
        "retention": ["90 days", "1 year", "7 years"],
        "abuse": ["brute force", "scraping", "enumeration"],
        "stage": ["CI", "pre-commit", "production"],
    },
    "mobile": {
        "intents": [
            "Build {screen} screen with {navigation} navigation",
            "Implement offline-first for {feature} with sync",
            "Add {animation} animation for {interaction}",
            "Optimize app size by {percentage}%",
            "Implement deep linking for {flow}",
            "Add biometric auth with {fallback} fallback",
            "Create widget for {data} display",
            "Implement push notifications with {targeting}",
            "Add crash reporting and analytics",
            "Optimize battery usage for {feature}",
        ],
        "screen": ["Home", "Detail", "Profile", "Settings", "Search", "Checkout"],
        "navigation": ["tab", "stack", "drawer", "bottom sheet"],
        "feature": ["cart", "messages", "content", "documents"],
        "animation": ["shared element", "gesture", "skeleton", "shimmer"],
        "interaction": ["pull-to-refresh", "swipe", "long press", "transition"],
        "percentage": ["20", "30", "40"],
        "flow": ["product", "checkout", "profile", "share"],
        "fallback": ["PIN", "password", "pattern"],
        "data": ["weather", "tasks", "notifications", "shortcuts"],
        "targeting": ["segment", "geo", "behavior", "personalized"],
    },
    "testing": {
        "intents": [
            "Set up {type} tests with {coverage}% coverage",
            "Create test fixtures for {component}",
            "Implement load testing for {scenario}",
            "Add visual regression for {pages}",
            "Create chaos engineering for {failure}",
            "Set up contract testing between {services}",
            "Implement mutation testing",
            "Add accessibility audit automation",
            "Create performance benchmarks for {operation}",
            "Build E2E tests with {framework}",
        ],
        "type": ["unit", "integration", "E2E", "contract", "property"],
        "coverage": ["80", "90", "95"],
        "component": ["API", "database layer", "auth", "payment"],
        "scenario": ["peak traffic", "sustained load", "spike", "soak"],
        "pages": ["critical path", "checkout", "all"],
        "failure": ["network", "database", "service", "region"],
        "services": ["frontend-backend", "microservices", "third-party"],
        "operation": ["API response", "query", "page load"],
        "framework": ["Playwright", "Cypress", "Selenium"],
    },
    "architecture": {
        "intents": [
            "Design {style} architecture for {scale} users",
            "Create migration plan from {current} to {target}",
            "Design event system with {guarantee} delivery",
            "Plan multi-region for {latency}ms latency",
            "Design caching with {layers} cache layers",
            "Create API versioning strategy",
            "Design for {availability} availability",
            "Plan sharding for {growth} growth",
            "Design async processing for {throughput}",
            "Create observability architecture",
        ],
        "style": ["microservices", "modular monolith", "serverless", "cell-based"],
        "scale": ["10K", "1M", "100M", "1B"],
        "current": ["monolith", "legacy", "on-prem"],
        "target": ["cloud-native", "microservices", "hybrid"],
        "guarantee": ["at-least-once", "exactly-once", "ordered"],
        "latency": ["50", "100", "200"],
        "layers": ["2", "3", "distributed"],
        "availability": ["99.9%", "99.99%", "99.999%"],
        "growth": ["10x", "100x", "1000x"],
        "throughput": ["10K/s", "100K/s", "1M/s"],
    },
    "database": {
        "intents": [
            "Optimize {query_type} query with {technique}",
            "Design schema for {use_case} with {pattern}",
            "Implement {replication} replication",
            "Set up connection pooling with {size} connections",
            "Create indexes for {access_pattern}",
            "Implement query result caching",
            "Design partitioning strategy for {table}",
            "Set up read replicas for {region}",
            "Implement soft deletes with archival",
            "Create database migration with zero downtime",
        ],
        "query_type": ["aggregation", "join", "full-text search", "geospatial"],
        "technique": ["indexing", "materialized view", "denormalization", "query rewrite"],
        "use_case": ["e-commerce", "social", "analytics", "time-series"],
        "pattern": ["star schema", "adjacency list", "nested set", "closure table"],
        "replication": ["sync", "async", "semi-sync"],
        "size": ["20", "50", "100"],
        "access_pattern": ["point lookup", "range scan", "prefix match"],
        "table": ["events", "orders", "logs", "users"],
        "region": ["us-east", "eu-west", "ap-south"],
    },
}

# =============================================================================
# STEP TEMPLATES - Much richer
# =============================================================================

EXPLORE_STEPS = {
    "research": [
        "Researching existing solutions for {topic}",
        "Investigating {aspect} to understand constraints",
        "Gathering requirements from {stakeholder}",
        "Analyzing trade-offs between {option_a} and {option_b}",
        "Exploring documentation for {technology}",
        "Testing hypothesis about {theory}",
        "Reviewing similar implementations in {codebase}",
        "Collecting metrics on current {behavior}",
    ],
    "clarify": [
        "Asking for clarification on {aspect}",
        "Confirming understanding of {requirement}",
        "Validating assumptions about {constraint}",
        "Seeking input on {decision}",
    ],
}

PLAN_STEPS = {
    "decompose": [
        "Breaking {task} into {count} sub-tasks",
        "Identifying dependencies between {component_a} and {component_b}",
        "Estimating complexity: {estimate}",
        "Creating technical approach for {feature}",
        "Defining acceptance criteria",
        "Planning database changes",
        "Mapping API contract",
    ],
    "design": [
        "Designing {component} architecture",
        "Creating schema for {entity}",
        "Planning error handling strategy",
        "Designing test approach",
    ],
}

EXECUTE_STEPS = {
    "implement": [
        "Implementing {component} logic",
        "Writing {type} tests for {feature}",
        "Connecting {service_a} to {service_b}",
        "Adding validation for {input}",
        "Refactoring {module} for {goal}",
        "Deploying to {environment}",
    ],
    "integrate": [
        "Integrating with {service}",
        "Setting up {tool} configuration",
        "Running integration tests",
        "Updating documentation",
    ],
}

REFLECT_STEPS = {
    "evaluate": [
        "Evaluating results of {action}",
        "Analyzing test failures in {area}",
        "Reconsidering approach after {finding}",
        "Assessing impact of {issue}",
    ],
    "recover": [
        "Recovering from {error}",
        "Implementing fallback for {failure}",
        "Rolling back {change}",
        "Adjusting plan based on {constraint}",
    ],
}

# =============================================================================
# FEEDBACK PATTERNS
# =============================================================================

SUCCESS_FEEDBACK = [
    {"success": True, "message": "Completed successfully"},
    {"success": True, "tests_passed": True, "coverage": "92%"},
    {"success": True, "deployed": True, "health": "ok"},
    {"success": True, "validated": True},
]

ERROR_FEEDBACK = [
    {"success": False, "error": "Connection timeout to {service}", "code": "ETIMEDOUT"},
    {"success": False, "error": "Rate limited, retry in {seconds}s", "code": "429"},
    {"success": False, "error": "Validation failed: {field} is required", "code": "400"},
    {"success": False, "error": "Permission denied for {resource}", "code": "403"},
    {"success": False, "error": "Dependency {package} not found", "code": "MODULE_NOT_FOUND"},
    {"success": False, "error": "Out of memory in {component}", "code": "OOM"},
    {"success": False, "error": "Deadlock detected", "code": "DEADLOCK"},
    {"success": False, "error": "Schema migration failed", "code": "MIGRATION_ERROR"},
    {"success": False, "error": "Tests failed: {count} failures", "code": "TEST_FAIL"},
    {"success": False, "error": "Build failed: {reason}", "code": "BUILD_ERROR"},
]

AMBIGUOUS_INTENTS = [
    "Fix it", "Make it work", "Do the thing", "You know what I mean",
    "Same as before but different", "Handle it", "The usual",
    "That bug we discussed", "The feature", "Make it better",
    "Clean it up", "Optimize it", "Just do it", "The issue",
    "That thing from the meeting", "What we talked about",
]

INTERRUPTIONS = [
    "URGENT: {issue} is down in production",
    "Stop - {blocker} is blocking the release",
    "Priority change: {task} is now P0",
    "Customer escalation: need {fix} immediately",
    "Security alert: {vulnerability} detected",
]


def fill(template: str, domain_key: str = None, custom: Dict = None) -> str:
    """Fill template with random values."""
    result = template
    fills = custom or {}
    
    # Add domain-specific fills
    if domain_key and domain_key in DOMAINS:
        for key, values in DOMAINS[domain_key].items():
            if key != "intents":
                fills[key] = values
    
    # Generic fills
    generic = {
        "topic": ["solution", "architecture", "approach", "implementation"],
        "aspect": ["requirements", "constraints", "scope", "timeline"],
        "stakeholder": ["product", "engineering", "customer", "security"],
        "option_a": ["option A", "current approach", "solution 1"],
        "option_b": ["option B", "new approach", "solution 2"],
        "technology": ["framework", "library", "service", "API"],
        "theory": ["root cause", "hypothesis", "assumption"],
        "codebase": ["main repo", "library", "similar project"],
        "behavior": ["performance", "reliability", "user flow"],
        "requirement": ["the scope", "expected behavior", "constraints"],
        "constraint": ["timeline", "resources", "technical limitation"],
        "decision": ["approach", "technology choice", "architecture"],
        "task": ["the feature", "implementation", "migration"],
        "count": ["3", "5", "7"],
        "component_a": ["frontend", "API", "database"],
        "component_b": ["backend", "cache", "queue"],
        "estimate": ["2 days", "1 week", "2 sprints"],
        "feature": ["the feature", "this component", "the system"],
        "component": ["service", "module", "handler"],
        "entity": ["user", "order", "transaction"],
        "type": ["unit", "integration", "E2E"],
        "service_a": ["frontend", "API", "worker"],
        "service_b": ["database", "cache", "queue"],
        "input": ["request", "payload", "parameters"],
        "module": ["auth", "core", "utils"],
        "goal": ["performance", "readability", "testability"],
        "environment": ["staging", "production", "dev"],
        "service": ["external API", "database", "third party"],
        "tool": ["Docker", "Terraform", "k8s"],
        "area": ["critical path", "edge cases", "integration"],
        "action": ["implementation", "deployment", "migration"],
        "finding": ["new constraint", "edge case", "blocker"],
        "issue": ["failure", "bug", "limitation"],
        "error": ["timeout", "crash", "validation error"],
        "failure": ["network", "database", "service"],
        "change": ["deployment", "migration", "refactor"],
        "seconds": ["30", "60", "300"],
        "field": ["email", "id", "name"],
        "resource": ["database", "API", "file"],
        "package": ["dependency", "module"],
        "blocker": ["production issue", "security bug", "data loss"],
        "vulnerability": ["SQL injection", "XSS", "RCE"],
        "fix": ["hotfix", "patch", "rollback"],
        "reason": ["syntax error", "type mismatch", "missing dependency"],
    }
    
    all_fills = {**generic, **fills}
    
    for key, values in all_fills.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values) if isinstance(values, list) else str(values), 1)
    
    return result


def mt_hash() -> str:
    return hashlib.sha256(str(random.random()).encode()).hexdigest()[:32]


class ProductionGenerator:
    """Production-grade generator targeting balanced 100K dataset for 30M model."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.domains = list(DOMAINS.keys())
        self.counts = {
            "exploration": 0, "planning": 0, "execution": 0,
            "reflection": 0, "adversarial": 0, "complex": 0,
        }
        self.stance_counts = {s.name: 0 for s in Stance}
        self.action_counts = {a.name: 0 for a in Action}
    
    def _stance(self, preferred: Stance) -> Stance:
        """Pick stance with balancing."""
        self.stance_counts[preferred.name] += 1
        return preferred
    
    def _action(self, preferred: Action) -> Action:
        """Track action distribution."""
        self.action_counts[preferred.name] += 1
        return preferred
    
    def _confidence(self, difficulty: float, certainty: str = "medium") -> float:
        """Calculate calibrated confidence."""
        base = {"high": 0.85, "medium": 0.65, "low": 0.35}[certainty]
        noise = (random.random() - 0.5) * 0.15
        return max(0.1, min(0.98, base + noise - difficulty * 0.1))
    
    def generate_exploration(self, difficulty: float) -> Scenario:
        """EXPLORE-focused scenario."""
        domain = random.choice(self.domains)
        num_steps = random.randint(5, 9)
        steps = []
        mth = mt_hash()
        
        # Start with vague/unclear intent
        if random.random() < 0.5:
            intent = random.choice(AMBIGUOUS_INTENTS)
        else:
            intent = f"I'm thinking about {random.choice(['how to', 'whether we should', 'the best way to'])} {fill(random.choice(DOMAINS[domain]['intents']), domain)[:60]}"
        
        for t in range(num_steps):
            if t == 0:
                # Initial exploration
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=intent,
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.3, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXPLORE),
                        confidence=self._confidence(difficulty, "low"),
                        action=self._action(Action.QUERY_USER if random.random() < 0.6 else Action.EXPLORE),
                        uncertainty=["scope", "requirements", "context"],
                    ),
                    reward=1.0,
                ))
            elif t < num_steps // 2:
                # More exploration
                category = random.choice(["research", "clarify"])
                step_template = random.choice(EXPLORE_STEPS[category])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[{"type": "explore", "step": t}],
                        feedback=random.choice(SUCCESS_FEEDBACK) if random.random() > 0.2 else None,
                        system={"urgency": 0.3, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXPLORE),
                        confidence=self._confidence(difficulty, "medium"),
                        action=self._action(random.choice([Action.EXPLORE, Action.QUERY_USER, Action.WRITE_MT])),
                        uncertainty=["completeness"],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps // 2:
                # Transition to plan
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="I think I understand now, let's plan the approach",
                        mt_hash=mth,
                        mt_delta=[{"type": "understanding_reached"}],
                        feedback=None,
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.PLAN),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t < num_steps - 1:
                # Execute
                step_template = random.choice(EXECUTE_STEPS["implement"])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.EXECUTE),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                # Complete
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Done",
                        mt_hash=mth,
                        mt_delta=[{"type": "complete"}],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.2, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.COMMIT),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["exploration"] += 1
        return Scenario(
            id=f"explore_{self.counts['exploration']:05d}",
            type=ScenarioType.PLANNING,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(domain=domain, complexity="exploration", required_pivots=0),
        )
    
    def generate_planning(self, difficulty: float) -> Scenario:
        """PLAN-focused with clear execution."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        intent = fill(random.choice(domain_data["intents"]), domain)
        
        num_steps = random.randint(4, 8)
        steps = []
        mth = mt_hash()
        
        for t in range(num_steps):
            if t == 0:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=intent,
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.DECOMPOSE_TASK if len(intent) > 50 else Action.PLAN),
                        uncertainty=["scope"] if difficulty > 0.5 else [],
                    ),
                    reward=1.0,
                ))
            elif t == 1:
                step_template = random.choice(PLAN_STEPS["decompose"])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[{"type": "plan"}],
                        feedback=None,
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.PLAN),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps - 1:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Complete and ship",
                        mt_hash=mth,
                        mt_delta=[{"type": "complete"}],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.3, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.COMMIT),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                step_template = random.choice(EXECUTE_STEPS["implement"] + EXECUTE_STEPS["integrate"])
                action = random.choice([Action.EXECUTE, Action.DELEGATE, Action.WRITE_MT])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[{"type": "progress", "step": t}],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(action),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["planning"] += 1
        return Scenario(
            id=f"plan_{self.counts['planning']:05d}",
            type=ScenarioType.PLANNING,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(domain=domain, complexity="planning", required_pivots=0),
        )
    
    def generate_reflection(self, difficulty: float) -> Scenario:
        """REFLECT-focused with errors/interruptions."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        intent = fill(random.choice(domain_data["intents"]), domain)
        
        num_steps = random.randint(6, 10)
        error_step = random.randint(2, num_steps - 4)
        steps = []
        mth = mt_hash()
        
        scenario_type = random.choice(["error_recovery", "interruption", "test_failure"])
        
        for t in range(num_steps):
            if t == 0:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=intent,
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.DECOMPOSE_TASK),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t < error_step:
                step_template = random.choice(EXECUTE_STEPS["implement"])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.EXECUTE),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t == error_step:
                # ERROR / INTERRUPTION / FAILURE
                if scenario_type == "interruption":
                    interrupt = fill(random.choice(INTERRUPTIONS))
                    steps.append(ScenarioStep(
                        t=t,
                        input_state=InputState(
                            intent=interrupt,
                            mt_hash=mt_hash(),
                            mt_delta=[{"type": "priority_change"}],
                            feedback=None,
                            system={"urgency": 0.95, "timestamp": 1704825600 + t * 300},
                        ),
                        expected_output=ExpectedOutput(
                            stance=self._stance(Stance.REFLECT),
                            confidence=self._confidence(difficulty, "low"),
                            action=self._action(Action.BACKTRACK),
                            uncertainty=["context_switch", "original_state"],
                        ),
                        reward=1.0,
                    ))
                else:
                    error = fill(random.choice([e["error"] for e in ERROR_FEEDBACK]))
                    steps.append(ScenarioStep(
                        t=t,
                        input_state=InputState(
                            intent="Continue",
                            mt_hash=mth,
                            mt_delta=[{"type": "error"}],
                            feedback={"success": False, "error": error},
                            system={"urgency": 0.8, "timestamp": 1704825600 + t * 300},
                        ),
                        expected_output=ExpectedOutput(
                            stance=self._stance(Stance.REFLECT),
                            confidence=self._confidence(difficulty, "low"),
                            action=self._action(Action.BACKTRACK),
                            uncertainty=["error_cause", "recovery_path"],
                        ),
                        reward=1.0,
                    ))
            elif t == error_step + 1:
                # Recovery
                step_template = random.choice(REFLECT_STEPS["recover"])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[{"type": "recovery"}],
                        feedback=None,
                        system={"urgency": 0.6, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "medium"),
                        action=self._action(Action.PLAN),
                        uncertainty=["alternative"],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps - 1:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Finally done",
                        mt_hash=mth,
                        mt_delta=[{"type": "complete"}],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.2, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.COMMIT),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                step_template = random.choice(EXECUTE_STEPS["implement"])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(step_template),
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.EXECUTE),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["reflection"] += 1
        return Scenario(
            id=f"reflect_{self.counts['reflection']:05d}",
            type=ScenarioType.REVISION,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain, complexity="reflection", required_pivots=1, error_recovery=True
            ),
        )
    
    def generate_adversarial(self, difficulty: float) -> Scenario:
        """Hard adversarial with ambiguity and edge cases."""
        domain = random.choice(self.domains)
        domain_data = DOMAINS[domain]
        
        num_steps = random.randint(8, 14)
        error_steps = [random.randint(3, num_steps - 5), random.randint(num_steps - 4, num_steps - 2)]
        if error_steps[0] >= error_steps[1]:
            error_steps[1] = error_steps[0] + 2
        
        steps = []
        mth = mt_hash()
        
        for t in range(num_steps):
            if t == 0:
                # Ambiguous start
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=random.choice(AMBIGUOUS_INTENTS),
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=None,
                        system={"urgency": 0.7, "timestamp": 1704825600},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXPLORE),
                        confidence=self._confidence(difficulty, "low"),
                        action=self._action(Action.QUERY_USER),
                        uncertainty=["intent", "scope", "context", "priority"],
                    ),
                    reward=1.0,
                ))
            elif t == 1:
                # Clarification
                clear_intent = fill(random.choice(domain_data["intents"]), domain)
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=f"Oh, I meant: {clear_intent}",
                        mt_hash=mth,
                        mt_delta=[{"type": "clarification"}],
                        feedback=None,
                        system={"urgency": 0.5, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.DECOMPOSE_TASK),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            elif t in error_steps:
                # Error
                error = fill(random.choice([e["error"] for e in ERROR_FEEDBACK]))
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Continue",
                        mt_hash=mth,
                        mt_delta=[{"type": "error"}],
                        feedback={"success": False, "error": error},
                        system={"urgency": 0.85, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.REFLECT),
                        confidence=self._confidence(difficulty, "low"),
                        action=self._action(Action.BACKTRACK),
                        uncertainty=["error_cause", "recovery"],
                    ),
                    reward=1.0,
                ))
            elif t - 1 in error_steps:
                # Recovery
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Trying alternative",
                        mt_hash=mth,
                        mt_delta=[{"type": "recovery"}],
                        feedback=None,
                        system={"urgency": 0.6, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.PLAN),
                        confidence=self._confidence(difficulty, "medium"),
                        action=self._action(Action.PLAN),
                        uncertainty=["alternative"],
                    ),
                    reward=1.0,
                ))
            elif t == num_steps - 1:
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent="Done",
                        mt_hash=mth,
                        mt_delta=[{"type": "complete"}],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.2, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(Action.COMMIT),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
            else:
                action = random.choice([Action.EXECUTE, Action.DELEGATE, Action.WAIT, Action.WRITE_MT])
                steps.append(ScenarioStep(
                    t=t,
                    input_state=InputState(
                        intent=fill(random.choice(EXECUTE_STEPS["implement"])),
                        mt_hash=mth,
                        mt_delta=[],
                        feedback=random.choice(SUCCESS_FEEDBACK),
                        system={"urgency": 0.4, "timestamp": 1704825600 + t * 300},
                    ),
                    expected_output=ExpectedOutput(
                        stance=self._stance(Stance.EXECUTE),
                        confidence=self._confidence(difficulty, "high"),
                        action=self._action(action),
                        uncertainty=[],
                    ),
                    reward=1.0,
                ))
        
        self.counts["adversarial"] += 1
        return Scenario(
            id=f"adv_{self.counts['adversarial']:05d}",
            type=ScenarioType.ADVERSARIAL,
            difficulty=difficulty,
            steps=steps,
            metadata=ScenarioMetadata(
                domain=domain, complexity="adversarial", required_pivots=2, 
                error_recovery=True, ambiguity_level=0.9
            ),
        )
    
    def generate(self, difficulty: float) -> Scenario:
        """Generate balanced scenario."""
        # Distribution targeting balanced stance output
        if difficulty < 0.3:
            weights = [0.35, 0.30, 0.20, 0.15]  # exploration, planning, reflection, adversarial
        elif difficulty < 0.6:
            weights = [0.25, 0.25, 0.25, 0.25]  # balanced
        else:
            weights = [0.15, 0.20, 0.30, 0.35]  # more adversarial/reflection
        
        scenario_type = random.choices(
            ["exploration", "planning", "reflection", "adversarial"],
            weights=weights, k=1
        )[0]
        
        if scenario_type == "exploration":
            return self.generate_exploration(difficulty)
        elif scenario_type == "planning":
            return self.generate_planning(difficulty)
        elif scenario_type == "reflection":
            return self.generate_reflection(difficulty)
        else:
            return self.generate_adversarial(difficulty)


def generate_production_dataset(
    output_dir: str,
    num_scenarios: int = 100000,
    seed: int = 42,
):
    """Generate production-grade dataset for 30M model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in ["planning", "revision", "adversarial"]:
        (output_path / subdir).mkdir(exist_ok=True)
    
    generator = ProductionGenerator(seed=seed)
    
    print(f"Generating {num_scenarios} PRODUCTION scenarios...")
    print(f"Domains: {len(DOMAINS)}")
    
    for i in range(num_scenarios):
        difficulty = min(1.0, (i / (num_scenarios * 0.6)) ** 0.8)  # Slower difficulty curve
        scenario = generator.generate(difficulty)
        
        subdir = scenario.type.value
        filename = f"{scenario.id}.json"
        scenario.save(str(output_path / subdir / filename))
        
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1}/{num_scenarios}")
            print(f"    Stances: {generator.stance_counts}")
    
    stats = {
        "total": num_scenarios,
        "scenario_counts": generator.counts,
        "stance_distribution": generator.stance_counts,
        "action_distribution": generator.action_counts,
        "domains": len(DOMAINS),
    }
    
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nProduction dataset complete!")
    print(f"  Scenarios: {generator.counts}")
    print(f"  Stances: {generator.stance_counts}")
    print(f"  Actions: {generator.action_counts}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PRODUCTION ODIN dataset")
    parser.add_argument("--output", type=str, default="../scenarios", help="Output dir")
    parser.add_argument("--num", type=int, default=100000, help="Num scenarios (default 100K)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    
    args = parser.parse_args()
    generate_production_dataset(args.output, args.num, args.seed)
