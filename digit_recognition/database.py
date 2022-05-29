""" Utilities and wrappers to handle access to the database

In most cases when you are using the database you should use a context manager.
If you need to hold a connection for longer you may use the connect and close methods.

    Typical Usage Example:
    >>> with ImageDatabase() as db:
    >>>     # Creating
    >>>     image = ImageEntry(7, data=b'...')
    >>>     db.add(image)
    >>>
    >>>     # Reading
    >>>     image2 = db.get(_id=1)
    >>>     all_images = db.all()
    >>>     training_images = db.filter(training=True)
    >>>     number_of_images = db.count()
    >>>
    >>>     # Update
    >>>     image = db.update(image, training=False)
    >>>
    >>>     # Delete
    >>>     db.remove(image)
"""

import hashlib
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple, Union
from uuid import uuid4

sqlite3.register_converter("boolean", lambda s: bool(int(s)))


class ImageEntry:
    """Represents a Image in the database.

    Should be used to create a new entry in the database.
    """
    def __init__(self, label: int, *, training: bool = True, custom: bool = True, uuid: Optional[str] = None,
                 created: Optional[datetime] = None, _id: Optional[int] = None, data: Optional[bytes] = None):
        if uuid is None:
            self.uuid = f"{label}{int(training)}{int(custom)}{uuid4()}"
        else:
            self.uuid = uuid

        if 0 > label > 9:
            raise ValueError("label must be between 0 and 9")

        if created is None:
            created = datetime.now()

        self.id = _id
        self.label = label
        self.training = training
        self.custom = custom
        self.created = created
        self.data = data

    def __repr__(self):
        return f"<Image uuid={self.uuid}>"

    def __str__(self):
        return str(self.path.resolve())

    @property
    def path(self) -> Path:
        """Returns the path to the linked file."""
        return ImageDatabase.get_path(self.uuid)

    def load(self):
        """Loads the image file into self.data."""
        with self.path.open("rb") as fp:
            self.data = fp.read()

    @classmethod
    def from_db_row(cls, row: sqlite3.Row):
        """Create a new ImageEntry from a Row returned from the database."""
        data = dict(row)
        mapping = {
            "_id": data.get("id"),
            "uuid": data.get("uuid"),
            "label": data.get("label"),
            "training": data.get("is_training"),
            "custom": data.get("is_custom"),
            "created": data.get("created")
        }
        return cls(**mapping)


class ImageDatabase:
    """High level wrapper to perform database operations.

    You can access direct access to the cursor to perform low level operations.
    """

    _path = Path(__file__).parent
    _images = _path / "digit-dataset"
    _db = _path / "images.db"

    @classmethod
    def get_path(cls, uuid: str) -> Path:
        """Return the path to an image with given uuid."""
        return cls._images / Path(uuid).with_suffix(".png")

    def __init__(self):
        """Create a new Instance of ImageDatabase.

        If the database does not exist, it will be created automatically.
        """
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

        self._images.mkdir(exist_ok=True)

        if not self._db.exists():
            with self:
                self._create_table()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def cursor(self) -> sqlite3.Cursor:
        """Exposes the low level database cursor."""
        if self._cursor is None:
            self._cursor = self._connection.cursor()
        return self._cursor

    @property
    def size(self) -> Tuple[int, int]:
        """Returns the file size that the database takes up.

        The size is composed of both the size of the database itself, and the size of the images on disk.
        These are respectively returned as a tuple.
        """
        image_size = sum(f.stat().st_size for f in self._images.glob("*") if f.is_file())
        db_size = self._db.stat().st_size
        return image_size, db_size

    def _create_table(self):
        """Creates the Image table for new databases.

        Also creates any indexes needed.
        """
        self.cursor.executescript("""
            create table Image
        (
            id integer primary key autoincrement,
            label integer not null,
            uuid text not null,
            hash text not null,
            is_training boolean not null,
            is_custom boolean not null,
            created timestamp not null
        );

        create index Image_label_index on Image (label);
        create unique index Image_hash_uindex on Image (hash);
        create unique index Image_uuid_uindex on Image (uuid);
        """)

    def _get_uuid(self,
                  image: Optional[ImageEntry] = None, *,
                  uuid: Optional[str] = None,
                  _id: Optional[int] = None,
                  strict: bool = False) -> Union[str, None]:
        """Helper function to allow multiple ways to specify a UUID."""
        if (image, uuid, _id).count(None) < 2:
            raise ValueError("To avoid ambiguity, only 1 lookup method may be provided.")

        if image is not None and image.uuid is not None:
            return image.uuid
        elif uuid is not None:
            return uuid
        elif _id is not None or (image is not None and image.id is not None):
            if _id is None:
                _id = image.id
            self.cursor.execute("SELECT uuid FROM Image where id = ?", (_id,))
            image = self.cursor.fetchone()
            if image is None:
                if strict:
                    raise ValueError("id not found")
                return None
            else:
                return image["uuid"]
        else:
            raise ValueError("No identifier provided. Must provide uuid or id.")

    def connect(self):
        """Connect to the database.

        This does not have to be called if you are using the context manager.
        """
        self._connection = sqlite3.connect(self._db, detect_types=sqlite3.PARSE_DECLTYPES)
        self._connection.row_factory = sqlite3.Row

    def close(self):
        """Close connection to the database.

        This does not have to be called if you are using the context manager.
        """
        self._connection.close()
        self._cursor = None

    def commit(self):
        """Commit all transactions to the database.

        Controlling when to commit can help to prevent performance issues.

        You do not usually need to call this manually unless you have specified `commit=False`
        in one of the non-idempotent commands.
        """
        self._connection.commit()

    def add(self, image: ImageEntry, commit=True):
        """Add an image to the database.

        if commit is False you will have to manually commit.
        """
        if image.data is None:
            raise ValueError("ImageEntry does not have any data")

        _hash = hashlib.sha1(image.data).hexdigest()

        query = "INSERT INTO Image (label, uuid, hash, is_training, is_custom, created) values (?, ?, ?, ?, ?, ?)"
        self.cursor.execute(query, (image.label, image.uuid, _hash, image.training, image.custom, image.created))
        if commit:
            self.commit()

        image.id = self.cursor.lastrowid

        with image.path.open("wb") as fp:
            fp.write(image.data)

    def remove(self,
               image: Optional[ImageEntry] = None, *,
               uuid: Optional[str] = None,
               _id: Optional[int] = None,
               commit: bool = True):
        """Remove an image from the database.

        if commit is False you will have to manually commit.
        """
        _uuid = self._get_uuid(image, uuid=uuid, _id=_id, strict=True)
        self.cursor.execute("DELETE FROM Image WHERE uuid = ?", (_uuid,))
        if commit:
            self.commit()

        self.get_path(_uuid).unlink(missing_ok=True)

    def get(self, image: Optional[ImageEntry] = None, *, uuid: Optional[str] = None, _id: Optional[int] = None):
        """Get a single image from the database."""
        _uuid = self._get_uuid(image, uuid=uuid, _id=_id)
        self.cursor.execute("SELECT * FROM Image WHERE uuid = ?", (_uuid,))

        image: sqlite3.Row = self.cursor.fetchone()
        if image is None:
            return None
        return ImageEntry.from_db_row(image)

    def update(self,
               image: Optional[ImageEntry] = None, *,
               uuid: Optional[str] = None,
               _id: Optional[int] = None,
               training: Optional[bool] = None,
               custom: Optional[bool] = None,
               commit: bool = True) -> Optional[ImageEntry]:
        """Update an image in the database.

        if commit is False you will have to manually commit.
        """
        _uuid = self._get_uuid(image, uuid=uuid, _id=_id)

        if training is None and custom is None:
            raise ValueError("No column is specified to update")

        query = '''
            UPDATE Image
            SET is_training = COALESCE(?, is_training),
                is_custom = COALESCE(?, is_custom)
            WHERE uuid = ?
        '''
        self.cursor.execute(query, (training, custom, _uuid))
        if commit:
            self.commit()

        return self.get(uuid=_uuid)

    def filter(self, *,
               label: Optional[int] = None,
               training: Optional[bool] = None,
               custom: Optional[bool] = None) -> Tuple[ImageEntry]:
        """Return all images which match given criteria.

        criteria are joined with an **AND** condition.
        """
        query = '''
                    SELECT * FROM Image WHERE (
                        (label = ?1 OR ?1 IS NULL)
                        AND (is_training = ?2 OR ?2 IS NULL)
                        AND (is_custom = ?3 OR ?3 IS NULL)
                    )
                '''
        self.cursor.execute(query, (label, training, custom))
        return tuple(map(ImageEntry.from_db_row, self.cursor.fetchall()))

    def all(self):
        """Return all images in database."""
        self.cursor.execute("SELECT * FROM Image")
        return tuple(map(ImageEntry.from_db_row, self.cursor.fetchall()))

    def count(self, *,
              label: Optional[int] = None,
              training: Optional[bool] = None,
              custom: Optional[bool] = None) -> int:
        """Return the number of images in database.

        Same parameters as filter().
        """
        query = '''
            SELECT COUNT(1) FROM Image WHERE (
                (label = ?1 OR ?1 IS NULL)
                AND (is_training = ?2 OR ?2 IS NULL)
                AND (is_custom = ?3 OR ?3 IS NULL)
            )
        '''
        self.cursor.execute(query, (label, training, custom))
        return int(self._cursor.fetchone()[0])

    def prune(self, dry=False):
        """Removes any corrupted files or database entries.

        i.e. if a file has no corresponding database entry or vice versa.

        Setting dry to True will prevent any modifications to the database.
        """
        db_uuids: Set[str] = set(map(lambda image: image.uuid, self.all()))
        file_uuids: Set[str] = set(map(lambda file: file.stem, self._images.glob("*.png")))

        no_files = db_uuids.difference(file_uuids)
        no_db = file_uuids.difference(db_uuids)

        for i in no_files:
            warnings.warn(f"Image {i} has no file")
            if not dry:
                self.remove(uuid=i, commit=False)
        self.commit()

        for i in no_db:
            warnings.warn(f"File {i}.png has no database entry")
            if not dry:
                self.get_path(i).unlink()
