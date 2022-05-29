#!/usr/bin/env python3

from typing import Iterator, Tuple

import click
import sys
from enum import Enum, Flag, auto

from PyQt6.QtWidgets import QApplication

from digit_recognition.database import ImageDatabase
from digit_recognition import DatasetApplication, RecognitionApplication


class ApplicationName(Enum):
    Recognition = "recognition"
    Dataset = "dataset"

    @classmethod
    def get_values(cls) -> Tuple[str]:
        members = cls.__members__.values()
        values: Iterator[str] = map(lambda i: i.value, members)
        return tuple(values)


class ImageType(Flag):
    CUSTOM = auto()
    NON_CUSTOM = auto()

    ALL = CUSTOM | NON_CUSTOM


@click.group()
def main():
    pass


@main.command()
@click.argument('name',
                default=ApplicationName.Recognition.value,
                type=click.Choice(ApplicationName.get_values(), case_sensitive=False))
def run(name: str):
    app = QApplication(sys.argv)
    app_name = ApplicationName(name)
    if app_name is ApplicationName.Recognition:
        RecognitionApplication(app).run()
    elif app_name is ApplicationName.Dataset:
        DatasetApplication(app).run()


@main.group()
def db():
    pass


@db.command()
@click.option('--dry', is_flag=True)
def prune(dry: bool):
    with ImageDatabase() as database:
        database.prune(dry)


@db.command()
@click.option('--all', 'image_type', flag_value=ImageType.ALL, default=True)
@click.option('--custom', 'image_type', flag_value=ImageType.CUSTOM)
@click.option('--non-custom', 'image_type', flag_value=ImageType.NON_CUSTOM)
def flush(image_type: ImageType):
    with ImageDatabase() as database:
        if image_type is ImageType.ALL:
            images = database.all()
        else:
            custom = ImageType.CUSTOM in image_type
            images = database.filter(custom=custom)
        for i in images:
            database.remove(i, commit=False)
        database.commit()


if __name__ == '__main__':
    main()
