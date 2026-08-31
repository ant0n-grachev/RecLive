from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_workflow(name: str) -> dict[str, object]:
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def test_ci_workflows_enforce_phase_one_required_gates() -> None:
    ci = load_workflow("ci.yml")
    security = load_workflow("security.yml")

    assert ci["on"] == {
        "push": {"branches": ["main", "hardening/reclive-security-data-trust"]},
        "pull_request": "",
    }
    assert ci["permissions"] == {"contents": "read"}
    ci_jobs = ci["jobs"]
    assert set(ci_jobs) == {"frontend", "backend"}

    frontend = ci_jobs["frontend"]
    assert frontend["runs-on"] == "ubuntu-latest"
    frontend_steps = frontend["steps"]
    assert frontend_steps == [
        {"uses": "actions/checkout@v4"},
        {
            "uses": "actions/setup-node@v4",
            "with": {"node-version": "22", "cache": "npm"},
        },
        {"run": "npm ci"},
        {"run": "npm run lint"},
        {
            "run": "npm run build",
            "env": {
                "VITE_API_BASE_URL": "http://127.0.0.1:8000",
                "VITE_SITE_URL": "http://127.0.0.1:4173",
            },
        },
        {"run": "npm run test:run"},
        {"run": "npx playwright install --with-deps chromium"},
        {
            "run": "npm run test:e2e -- tests/e2e/route-smoke.spec.ts",
            "env": {
                "VITE_API_BASE_URL": "http://127.0.0.1:8000",
                "VITE_SITE_URL": "http://127.0.0.1:4173",
            },
        },
    ]

    backend = ci_jobs["backend"]
    assert backend["runs-on"] == "ubuntu-latest"
    assert backend["services"] == {
        "mysql": {
            "image": "mysql:8.4",
            "env": {
                "MYSQL_DATABASE": "reclive_test",
                "MYSQL_USER": "reclive",
                "MYSQL_PASSWORD": "reclive-ci-password",
                "MYSQL_ROOT_PASSWORD": "root-ci-password",
            },
            "ports": ["3306:3306"],
            "options": '--health-cmd="mysqladmin ping -h 127.0.0.1 -uroot -proot-ci-password" --health-interval=10s --health-timeout=5s --health-retries=10',
        }
    }
    assert backend["steps"] == [
        {"uses": "actions/checkout@v4"},
        {
            "uses": "actions/setup-python@v5",
            "with": {"python-version": "3.12", "cache": "pip"},
        },
        {"run": "python -m pip install --upgrade pip"},
        {
            "run": "python -m pip install -r server/requirements.txt -r server/requirements-dev.txt"
        },
        {"run": "ruff check server tests"},
        {
            "run": "python -m pytest -q",
            "env": {
                "TEST_MYSQL_HOST": "127.0.0.1",
                "TEST_MYSQL_PORT": "3306",
                "TEST_MYSQL_USER": "reclive",
                "TEST_MYSQL_PASSWORD": "reclive-ci-password",
                "TEST_MYSQL_DATABASE": "reclive_test",
                "TEST_MYSQL_ADMIN_USER": "root",
                "TEST_MYSQL_ADMIN_PASSWORD": "root-ci-password",
            },
        },
    ]

    assert security["on"] == ci["on"]
    assert security["permissions"] == {"contents": "read"}
    security_jobs = security["jobs"]
    assert set(security_jobs) == {"gitleaks", "dependency-review"}

    gitleaks = security_jobs["gitleaks"]
    assert gitleaks["runs-on"] == "ubuntu-latest"
    assert gitleaks["steps"] == [
        {"uses": "actions/checkout@v6", "with": {"fetch-depth": "0"}},
        {
            "name": "Scan all reachable history with redacted output",
            "run": 'docker run --rm -v "$PWD:/repo" zricethezav/gitleaks:v8.28.0 git --redact --log-opts="--all" /repo',
        },
    ]

    dependency_review = security_jobs["dependency-review"]
    assert dependency_review["if"] == "github.event_name == 'pull_request'"
    assert dependency_review["runs-on"] == "ubuntu-latest"
    assert dependency_review["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert dependency_review["steps"] == [
        {"uses": "actions/checkout@v6"},
        {"uses": "actions/dependency-review-action@v4"},
    ]


def test_dependabot_tracks_npm_pip_and_github_actions_weekly() -> None:
    dependabot = yaml.load(
        (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )

    assert dependabot["version"] == "2"
    assert len(dependabot["updates"]) == 3
    assert {
        (
            update["package-ecosystem"],
            update["directory"],
            update["schedule"]["interval"],
        )
        for update in dependabot["updates"]
    } == {
        ("npm", "/", "weekly"),
        ("pip", "/server", "weekly"),
        ("github-actions", "/", "weekly"),
    }
