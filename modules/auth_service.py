# modules/auth_service.py

from supabase_client import supabase


# ============================================================
# REGISTER (Supabase handles email verification)
# ============================================================

def register_user(email: str, password: str):
    try:
        res = supabase.auth.sign_up({
            "email": email,
            "password": password
        })

        return True, "📧 Verification email sent! Please check your inbox."

    except Exception as e:
        return False, str(e)


# ============================================================
# LOGIN (only verified users allowed)
# ============================================================

def login_user(email: str, password: str):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if res.user:
            return res.user.id, ""

        return None, "Login failed"

    except Exception as e:
        err = str(e)

        if "Email not confirmed" in err:
            return None, "⚠️ Please verify your email before logging in."

        return None, err


# ============================================================
# GET CURRENT USER
# ============================================================

def get_current_user():
    try:
        res = supabase.auth.get_user()
        return res.user
    except:
        return None


# ============================================================
# LOGOUT
# ============================================================

def logout_user():
    try:
        supabase.auth.sign_out()
    except:
        pass