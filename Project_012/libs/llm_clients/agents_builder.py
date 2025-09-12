from __future__ import annotations
import re, yaml, pandas as pd
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass
from utils.retry import retry_on_error

def _lazy_sqlalchemy():
    from sqlalchemy import create_engine, text, inspect
    return create_engine, text, inspect

# ---------- Base ----------
@dataclass
class BaseAgent:
    name: str
    kind: str
    config: Dict[str, Any]
    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.kind, **self.config}

# ---------- LLM-driven SQL Agent ----------
class SQLLlmAgent(BaseAgent):
    """
    LLM-driven SQL agent:
      • Uses llm_factory(cfg) to build an LLMChain-like object with .invoke(dict)->str (SQL).
      • Introspects DB schema and injects it as {schema} unless provided.
      • Executes ONLY a single SELECT on conn_str and returns a pandas DataFrame.
      • Retries (empty/error) up to 3 times, feeding error back to prompt as {error_context}.
    """
    def __init__(self, name: str, kind: str, config: Dict[str, Any], llm_factory: Callable[[Dict[str, Any]], Any]):
        super().__init__(name, kind, config)
        create_engine, _, _ = _lazy_sqlalchemy()
        self._engine = create_engine(config["conn_str"])
        self._llm = llm_factory(config)
        self._schema_cache: Optional[str] = None
    
    def get_schema_json(
        self,
        *,
        refresh: bool = False,
        include_types: bool = True,
        include_foreign_keys: bool = True,
        sample_rows: int = 0,
        max_cols: int = 200
    ) -> Dict[str, Any]:
        """
        Returns a structured JSON schema:
        {
          "tables": {
            "<table>": {
              "columns": [{"name": "...", "type": "...", "primary_key": bool, "nullable": bool, "default": "..."}],
              "primary_keys": ["..."],
              "foreign_keys": [{"columns": ["..."], "referred_table": "...", "referred_columns": ["..."]}],
              "samples": [ {col: value, ...}, ... ]   # optional
            }, ...
          }
        }
        Cached (without samples) unless refresh=True.
        """
        if self._schema_cache_json is not None and not refresh and sample_rows == 0:
            return self._schema_cache_json

        _, _, inspect = _lazy_sqlalchemy()
        insp = inspect(self._engine)
        out: Dict[str, Any] = {"tables": {}}

        for tbl in insp.get_table_names():
            cols = insp.get_columns(tbl)
            tinfo: Dict[str, Any] = {"columns": [], "primary_keys": [], "foreign_keys": []}
            for c in cols[:max_cols]:
                entry = {
                    "name": c["name"],
                    "primary_key": bool(c.get("primary_key")),
                    "nullable": bool(c.get("nullable", True)),
                }
                if include_types:
                    entry["type"] = str(c.get("type"))
                if c.get("default") is not None:
                    entry["default"] = str(c.get("default"))
                tinfo["columns"].append(entry)
                if entry["primary_key"]:
                    tinfo["primary_keys"].append(entry["name"])

            if include_foreign_keys:
                try:
                    fks = insp.get_foreign_keys(tbl)
                    for fk in fks:
                        tinfo["foreign_keys"].append({
                            "columns": fk.get("constrained_columns", []) or [],
                            "referred_table": fk.get("referred_table"),
                            "referred_columns": fk.get("referred_columns", []) or []
                        })
                except Exception:
                    pass

            if sample_rows > 0:
                try:
                    with self._engine.connect() as conn:
                        df_sample = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT {int(sample_rows)}", conn)
                    # make JSON-serializable
                    samples = []
                    for _, row in df_sample.iterrows():
                        samples.append({k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()})
                    tinfo["samples"] = samples
                except Exception:
                    tinfo["samples"] = []

            out["tables"][tbl] = tinfo

        if sample_rows == 0:
            self._schema_cache_json = out
        return out

    # ----- Schema -----
    def get_db_schema(self, *, refresh: bool = False, sample_rows: int = 0, max_cols: int = 40) -> str:
        if self._schema_cache is not None and not refresh and sample_rows == 0:
            return self._schema_cache

        _, _, inspect = _lazy_sqlalchemy()
        insp = inspect(self._engine)
        lines: List[str] = ["# Database Schema"]
        for tbl in insp.get_table_names():
            cols = insp.get_columns(tbl)
            pks  = [c["name"] for c in cols if c.get("primary_key")]
            col_lines = []
            for c in cols[:max_cols]:
                col_lines.append(f"- {c['name']} ({str(c.get('type'))})" + (" [PK]" if c.get("primary_key") else ""))
            if len(cols) > max_cols:
                col_lines.append(f"- ... ({len(cols)-max_cols} more columns omitted)")
            lines.append(f"\n## {tbl}")
            if pks:
                lines.append(f"**Primary Key:** {', '.join(pks)}")
            lines.append("**Columns:**")
            lines.extend(col_lines)
            if sample_rows and sample_rows > 0:
                try:
                    with self._engine.connect() as conn:
                        df_sample = pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT {int(sample_rows)}", conn)
                    lines.append("**Sample Rows:**")
                    for _, r in df_sample.iterrows():
                        preview = ", ".join(f"{k}={repr(v)}" for k, v in r.items())
                        lines.append(f"- {preview}")
                except Exception:
                    pass
        schema_md = "\n".join(lines)
        if sample_rows == 0:
            self._schema_cache = schema_md
        return schema_md

    # ----- SQL helpers -----
    @staticmethod
    def _extract_sql(text: str) -> str:
        fence = re.findall(r"```(?:sql)?\s*(.+?)\s*```", text, flags=re.I | re.S)
        sql = fence[0].strip() if fence else text.strip()
        if ";" in sql:
            sql = sql.split(";")[0].strip()
        return sql

    @staticmethod
    def _is_safe_select(sql: str) -> bool:
        s = sql.strip().rstrip(";").lower()
        if not s.startswith("select"):
            return False
        forbidden = r"\b(insert|update|delete|drop|alter|create|truncate|merge|grant|revoke)\b"
        return re.search(forbidden, s) is None

    def run(self, **vars) -> str:
        sql_text = self._llm.invoke(vars)
        return self._extract_sql(sql_text)

    # ----- Main API (with retry) -----
    @retry_on_error(max_attempts=3)
    def query(self, *, include_schema: bool = True, schema_refresh: bool = False,
              schema_sample_rows: int = 0, **vars) -> pd.DataFrame:
        """
        End-to-end: (optional schema) + LLM → SQL → execute → DataFrame.
        - include_schema: injects {schema} var if not present
        - schema_refresh: force re-introspect
        - schema_sample_rows: include N sample rows per table in schema text
        - error_context: (injected by retry decorator) prior failure details
        """
        from sqlalchemy import text as _text
        if include_schema and "schema" not in vars:
            vars = {**vars, "schema": self.get_db_schema(refresh=schema_refresh, sample_rows=schema_sample_rows)}
        # error_context may be present (set by decorator on retries)
        if "error_context" not in vars:
            vars = {**vars, "error_context": ""}

        sql = self.generate_sql(**vars)
        if not self._is_safe_select(sql):
            raise ValueError(f"Refusing non-SELECT or unsafe SQL:\n{sql}")

        with self._engine.begin() as conn:
            return pd.read_sql_query(_text(sql), conn, params=vars)

# ---------- Plain LLM Agent ----------
class LLMAgent(BaseAgent):
    def __init__(self, name: str, kind: str, config: Dict[str, Any], llm_factory: Optional[Callable[[Dict[str, Any]], Any]]):
        super().__init__(name, kind, config)
        self._llm = llm_factory(config) if llm_factory else None
    def run(self, prompt: Optional[str] = None, **vars):
        if hasattr(self._llm, "invoke"):
            inputs = {"input": prompt} if (prompt and not vars) else vars
            out = self._llm.invoke(inputs)
            return {"text": out, "raw": out}
        return self._llm(prompt=prompt, **vars)

# ---------- Loader ----------
class AgentWrapper:
    def __init__(self, yaml_path_or_dict: str | Dict[str, Any], *, llm_factory: Callable[[Dict[str, Any]], Any]):
        cfg = yaml.safe_load(open(yaml_path_or_dict, "r", encoding="utf-8")) if isinstance(yaml_path_or_dict, str) else yaml_path_or_dict
        if not isinstance(cfg, dict) or "agents" not in cfg:
            raise ValueError("YAML must have top-level 'agents' list")
        self.agents_cfg: List[Dict[str, Any]] = cfg["agents"]
        self.llm_factory = llm_factory

    def build_all(self) -> Dict[str, BaseAgent]:
        agents: Dict[str, BaseAgent] = {}
        for acfg in self.agents_cfg:
            name = acfg["name"]; typ = acfg["type"].lower()
            if typ == "sql":
                agents[name] = SQLLlmAgent(name=name, kind="sql_llm", config=acfg, llm_factory=self.llm_factory)
            elif typ == "llm_agent":
                agents[name] = LLMAgent(name=name, kind="llm_agent", config=acfg, llm_factory=self.llm_factory)
            else:
                raise ValueError(f"Unknown agent type: {typ}")
        return agents

    def build_one(self, agent_name: str) -> BaseAgent:
        """Build a single agent by name"""
        for acfg in self.agents_cfg:
            if acfg["name"] == agent_name:
                name = acfg["name"]; typ = acfg["type"].lower()
                if typ == "sql":
                    return SQLLlmAgent(name=name, kind="sql_llm", config=acfg, llm_factory=self.llm_factory)
                elif typ == "llm_agent":
                    return LLMAgent(name=name, kind="llm_agent", config=acfg, llm_factory=self.llm_factory)
                else:
                    raise ValueError(f"Unknown agent type: {typ}")
        raise ValueError(f"Agent '{agent_name}' not found in configuration")
