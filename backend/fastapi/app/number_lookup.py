import json
import os
import re
from http.client import HTTPSConnection
from threading import BoundedSemaphore
from urllib.parse import urlencode

from .errors import ServiceError
from .schemas import NumberLookupResult


SLOTS = BoundedSemaphore(2)
RESPONSE_LIMIT = 64 * 1024


def verify_number(number, key):
    connection = HTTPSConnection("api.veriphone.io", timeout=5)
    try:
        connection.request("POST", "/v3/verify",
                           body=urlencode({"phone": number, "mode": "static", "record": "false"}),
                           headers={"Authorization": f"Bearer {key}",
                                    "Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
        response = connection.getresponse()
        if response.status in {401, 402, 403, 429, 503}:
            raise ServiceError("NUMBER_LOOKUP_UNAVAILABLE", "Number lookup is currently unavailable.")
        if response.status != 200:
            raise ValueError("Provider failed.")
        body = response.read(RESPONSE_LIMIT + 1)
        if len(body) > RESPONSE_LIMIT:
            raise ValueError("Provider response exceeds limit.")
        return json.loads(body)
    except ServiceError:
        raise
    except (OSError, TimeoutError):
        raise ServiceError("NUMBER_LOOKUP_UNAVAILABLE", "Number lookup is currently unavailable.") from None
    except Exception:
        raise ServiceError("NUMBER_LOOKUP_FAILED", "Number lookup could not be completed.", 502) from None
    finally:
        connection.close()


def normalize_result(original, value):
    if not isinstance(value, dict) or value.get("status") != "success" or type(value.get("phone_valid")) is not bool:
        raise ValueError("Invalid provider response.")
    valid = value["phone_valid"]
    fields = {}
    for target, source in (("e164", "e164"), ("country_name", "country"), ("country_code", "country_code"),
                           ("carrier", "carrier"), ("line_type", "phone_type")):
        item = value.get(source)
        if item is not None and (not isinstance(item, str) or len(item) > 200):
            raise ValueError("Invalid provider metadata.")
        fields[target] = item or None
    if valid and (not fields["e164"] or not re.fullmatch(r"\+[1-9][0-9]{6,14}", fields["e164"])):
        raise ValueError("Invalid provider number.")
    reasons = {"too_short": "TOO_SHORT", "too_long": "TOO_LONG", "invalid_length": "INVALID_LENGTH",
               "invalid_country_code": "INVALID_COUNTRY_CODE", "unrecognized_range": "UNRECOGNIZED_RANGE",
               "not_a_number": "NOT_A_NUMBER"}
    return NumberLookupResult(input=original, status="looked_up", valid=valid,
                              reason_code=None if valid else reasons.get(value.get("reason"), "INVALID_NUMBER"), **fields)


def lookup_numbers(numbers):
    key = os.environ.get("VERIPHONE_API_KEY", "").strip()
    if not key or not SLOTS.acquire(blocking=False):
        raise ServiceError("NUMBER_LOOKUP_UNAVAILABLE", "Number lookup is currently unavailable.")
    completed = 0
    try:
        results = []
        for original in numbers:
            number = re.sub(r"[\s().-]", "", original)
            if number.startswith("00"):
                number = "+" + number[2:]
            digits = number.lstrip("+")
            if not re.fullmatch(r"\+?[0-9]+", number):
                results.append(NumberLookupResult(input=original, status="not_a_number", valid=False, reason_code="NOT_A_NUMBER"))
            elif not number.startswith("+") and len(digits) <= 6:
                results.append(NumberLookupResult(input=original, status="not_a_number", valid=False, reason_code="SHORT_CODE"))
            elif not number.startswith("+"):
                results.append(NumberLookupResult(input=original, status="unavailable", reason_code="COUNTRY_CODE_REQUIRED"))
            elif not re.fullmatch(r"\+[1-9][0-9]{6,14}", number):
                results.append(NumberLookupResult(input=original, status="not_a_number", valid=False, reason_code="INVALID_FORMAT"))
            else:
                results.append(normalize_result(original, verify_number(number, key)))
                completed += 1
        return results
    except ServiceError as error:
        if completed:
            raise ServiceError("NUMBER_LOOKUP_FAILED", "Number lookup could not be completed.", error.status_code) from None
        raise
    except Exception:
        raise ServiceError("NUMBER_LOOKUP_FAILED", "Number lookup could not be completed.", 502) from None
    finally:
        SLOTS.release()
