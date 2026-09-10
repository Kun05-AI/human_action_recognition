from dataclasses import dataclass


@dataclass
class Identity:
    display_id: int
    name: str


class IdentityManager:
    def __init__(self) -> None:
        self._tracker_to_identity: dict[int, Identity] = {}
        self._next_display_id = 1

    def get_identity(self, tracker_id: int) -> Identity:
        if tracker_id not in self._tracker_to_identity:
            display_id = self._next_display_id

            identity = Identity(
                display_id=display_id,
                name=self._generate_name(display_id),
            )

            self._tracker_to_identity[tracker_id] = identity
            self._next_display_id += 1

        return self._tracker_to_identity[tracker_id]

    @staticmethod
    def _generate_name(display_id: int) -> str:
        """
        1 -> a
        2 -> b
        ...
        26 -> z
        27 -> aa
        28 -> ab
        """
        number = display_id
        result = ""

        while number > 0:
            number -= 1
            result = chr(ord("a") + (number % 26)) + result
            number //= 26

        return result

    def get_all_identities(self) -> dict[int, Identity]:
        return dict(self._tracker_to_identity)

    def reset(self) -> None:
        self._tracker_to_identity.clear()
        self._next_display_id = 1