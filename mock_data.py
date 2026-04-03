"""
Mock database for Bookly customer support agent.
Simulates orders, customers, and store policies.
"""

import re
from datetime import date, timedelta

TODAY = date(2026, 4, 1)

ORDER_ID_PATTERN = re.compile(r"^ORD-\d+$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_order_id(order_id: str) -> str | None:
    """Normalize and validate order ID format. Returns normalized ID or None."""
    order_id = order_id.strip().upper()
    if not ORDER_ID_PATTERN.match(order_id):
        return None
    return order_id


def _validate_email(email: str) -> str | None:
    """Normalize and validate email format. Returns normalized email or None."""
    email = email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        return None
    return email

CUSTOMERS = {
    "C-1001": {
        "id": "C-1001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "membership": "premium",
    },
    "C-1002": {
        "id": "C-1002",
        "name": "Bob Martinez",
        "email": "bob@example.com",
        "membership": "standard",
    },
    "C-1003": {
        "id": "C-1003",
        "name": "Carol Chen",
        "email": "carol@example.com",
        "membership": "standard",
    },
}

ORDERS = {
    "ORD-5001": {
        "order_id": "ORD-5001",
        "customer_id": "C-1001",
        "items": [
            {"title": "The Great Gatsby", "qty": 1, "price": 12.99},
            {"title": "1984", "qty": 1, "price": 9.99},
        ],
        "status": "delivered",
        "order_date": str(TODAY - timedelta(days=10)),
        "delivery_date": str(TODAY - timedelta(days=3)),
        "shipping_method": "standard",
        "total": 22.98,
        "tracking_number": "TRK-88234A",
    },
    "ORD-5002": {
        "order_id": "ORD-5002",
        "customer_id": "C-1001",
        "items": [
            {"title": "Dune", "qty": 1, "price": 15.99},
        ],
        "status": "shipped",
        "order_date": str(TODAY - timedelta(days=2)),
        "estimated_delivery": str(TODAY + timedelta(days=3)),
        "shipping_method": "express",
        "total": 15.99,
        "tracking_number": "TRK-99102B",
    },
    "ORD-5003": {
        "order_id": "ORD-5003",
        "customer_id": "C-1002",
        "items": [
            {"title": "To Kill a Mockingbird", "qty": 1, "price": 11.49},
            {"title": "Pride and Prejudice", "qty": 2, "price": 8.99},
        ],
        "status": "processing",
        "order_date": str(TODAY - timedelta(days=1)),
        "estimated_delivery": str(TODAY + timedelta(days=6)),
        "shipping_method": "standard",
        "total": 29.47,
        "tracking_number": None,
    },
    "ORD-5004": {
        "order_id": "ORD-5004",
        "customer_id": "C-1003",
        "items": [
            {"title": "The Hobbit", "qty": 1, "price": 14.99},
        ],
        "status": "delivered",
        "order_date": str(TODAY - timedelta(days=45)),
        "delivery_date": str(TODAY - timedelta(days=38)),
        "shipping_method": "standard",
        "total": 14.99,
        "tracking_number": "TRK-77019C",
    },
}

POLICIES = {
    "returns": (
        "Bookly accepts returns within 30 days of delivery for a full refund. "
        "Books must be in original condition (no writing, highlighting, or damage). "
        "Digital purchases and sale items are final sale and cannot be returned. "
        "Return shipping is free for Premium members; standard members pay a flat $4.99 return shipping fee."
    ),
    "shipping": (
        "Standard shipping: 5-7 business days, free on orders over $25. "
        "Express shipping: 2-3 business days, $7.99 flat rate. "
        "Overnight shipping: next business day, $14.99 flat rate. "
        "Premium members get free express shipping on all orders."
    ),
    "password_reset": (
        "Customers can reset their password by clicking 'Forgot Password' on the login page. "
        "A reset link is sent to the email on file and expires after 1 hour. "
        "If the customer doesn't receive the email, verify the email address and check spam. "
        "Support agents can trigger a password reset email but cannot view or set passwords directly."
    ),
    "membership": (
        "Bookly offers two tiers: Standard (free) and Premium ($9.99/month). "
        "Premium benefits: free express shipping, free return shipping, early access to sales, "
        "and 10% discount on all orders. Members can upgrade or cancel anytime from Account Settings."
    ),
}


def lookup_order(order_id: str) -> dict | None:
    """Look up an order by ID."""
    order_id = _validate_order_id(order_id)
    if not order_id:
        return {"error": "Invalid order ID format. Expected format: ORD-XXXX (e.g., ORD-5001)."}
    return ORDERS.get(order_id)


def lookup_orders_by_email(email: str) -> list[dict] | dict:
    """Find all orders for a customer email."""
    email = _validate_email(email)
    if not email:
        return {"error": "Invalid email format. Please provide a valid email address."}
    customer = next((c for c in CUSTOMERS.values() if c["email"] == email), None)
    if not customer:
        return []
    return [o for o in ORDERS.values() if o["customer_id"] == customer["id"]]


def initiate_return(order_id: str, reason: str) -> dict:
    """Process a return request. Returns result with status."""
    order_id = _validate_order_id(order_id)
    if not order_id:
        return {"success": False, "error": "Invalid order ID format. Expected format: ORD-XXXX (e.g., ORD-5001)."}
    order = ORDERS.get(order_id)
    if not order:
        return {"success": False, "error": "Order not found."}

    if order["status"] != "delivered":
        return {"success": False, "error": f"Order status is '{order['status']}'. Only delivered orders can be returned."}

    # Check 30-day return window
    delivery = date.fromisoformat(order["delivery_date"])
    if (TODAY - delivery).days > 30:
        return {"success": False, "error": "Return window has closed. Returns must be initiated within 30 days of delivery."}

    customer = CUSTOMERS.get(order["customer_id"], {})
    return_fee = 0.00 if customer.get("membership") == "premium" else 4.99

    return {
        "success": True,
        "return_id": f"RET-{order_id.split('-')[1]}",
        "refund_amount": order["total"] - return_fee,
        "return_shipping_fee": return_fee,
        "instructions": "A prepaid return label has been emailed. Please ship the book(s) within 7 days.",
    }


def send_password_reset(email: str) -> dict:
    """Trigger a password reset email."""
    email = _validate_email(email)
    if not email:
        return {"success": False, "error": "Invalid email format. Please provide a valid email address."}
    customer = next((c for c in CUSTOMERS.values() if c["email"] == email), None)
    if not customer:
        return {"success": False, "error": "No account found with that email address."}
    return {
        "success": True,
        "message": f"Password reset email sent to {email}. The link expires in 1 hour.",
    }


def get_policy(topic: str) -> str | None:
    """Retrieve a store policy by topic."""
    return POLICIES.get(topic.lower())


def escalate_to_human(reason: str, summary: str) -> dict:
    """Escalate the conversation to a human support agent."""
    return {
        "success": True,
        "ticket_id": "ESC-7042",
        "message": "Ok, let's get some extra help! I've placed you in the queue for a human agent. Estimated wait time: ~3 minutes.",
        "reason": reason,
        "summary": summary,
    }
