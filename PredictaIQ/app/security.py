"""Admin router'ı için basit paylaşımlı-anahtar (shared secret) auth.

`/api/v1/admin/*` altındaki endpoint'ler (model eğitimi, canlı veri
senkronizasyonu) hem pahalı (CPU/DB) hem de veri bütünlüğünü etkileyen
işlemler -- eskiden tamamen açık (auth'suz) haldeydi. GitHub Actions
cron'unun bu endpoint'leri çağırabilmesi ve başka kimsenin çağıramaması
için `X-Admin-Api-Key` header'ı bekleniyor.
"""
from fastapi import Header, HTTPException

from app.config import settings


async def verify_admin_key(x_admin_api_key: str = Header(default="")) -> None:
    if not settings.admin_api_key:
        # .env'de tanımlanmamışsa (örn. yerel geliştirme) auth'u atla --
        # ama production'da bu değişkenin MUTLAKA set edilmesi gerekir.
        return
    if x_admin_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Geçersiz veya eksik X-Admin-Api-Key")
