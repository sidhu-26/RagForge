"""initial_metadata

Revision ID: 405bd1185000
Revises:
Create Date: 2026-10-22 13:31:53.264247

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "405bd1185000"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
