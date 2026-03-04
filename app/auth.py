from __future__ import annotations

import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

bearer_scheme = HTTPBearer()


def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    token = credentials.credentials
    secret = os.environ["SUPABASE_JWT_SECRET"]
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )
    return payload
