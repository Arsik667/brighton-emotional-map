"""Единая настройка логов для всех скриптов."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """
    Включить понятные логи.

    Почему logging, а не print: логи можно отключить одним флагом,
    у них есть уровни важности и время события. print в библиотечном
    коде - признак того, что автор не думал о том, как это запускать.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Библиотеки любят шуметь на уровне INFO - приглушаем.
    for noisy in ("urllib3", "httpx", "httpcore", "huggingface_hub",
                  "filelock", "transformers", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
