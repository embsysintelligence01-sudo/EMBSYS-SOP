from dataclasses import dataclass
import os
from pathlib import Path
from app.config import tomllib

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    runtime: Path = ROOT/'runtime/production'
    mode: str = 'production'
    hardware: str = 'virtual'
    host: str = '127.0.0.1'
    port: int = 8765
    token: str = ''
    cpu_threads: int = 2
    max_hosts: int = 2
    max_background: int = 2
    model_cache_seconds: int = 90
    enabled: tuple = ()
    replay_source: str = ''

    @classmethod
    def load(cls, **overrides):
        path=ROOT/'configs/production.toml'
        values=tomllib.loads(path.read_text()) if path.exists() else {}
        values.update({k:v for k,v in overrides.items() if v is not None})
        values['token']=os.environ.get('EMBSYS_API_TOKEN', values.get('token',''))
        result=cls(**values);result.runtime=Path(result.runtime).resolve()
        if result.host not in ('127.0.0.1','localhost','::1') and len(result.token)<32:
            raise ValueError('LAN access requires EMBSYS_API_TOKEN with at least 32 characters')
        if result.cpu_threads<1 or result.max_hosts<1 or result.max_background<1:
            raise ValueError('Resource budgets must be positive')
        return result
