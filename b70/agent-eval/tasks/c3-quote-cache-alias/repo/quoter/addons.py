"""Add-ons: extra lines on top of a plan."""
from . import catalog


class AddonError(ValueError):
    pass


def attach(lines, plan, skus):
    """Add the lines for the add-ons `skus` to `lines`, a quote on `plan`.

    Every add-on must be sold on that plan.
    """
    for sku in skus:
        addon = catalog.get_addon(sku)
        if plan not in addon["plans"]:
            raise AddonError(f"add-on {sku} is not sold on the {plan} plan")
        lines.append(addon)
    return lines
