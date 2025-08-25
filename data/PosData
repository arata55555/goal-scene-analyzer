import typing
import itertools
from typing import Tuple, List, Union

Pos = Tuple[float, float]
TeamPos = List[Pos]

class PosData:
    def __init__(self, ball: Pos, left: TeamPos, right: TeamPos) -> None:
        """
        PosDataオブジェクトを初期化します。
        
        Parameters
        ----------
        ball : Pos
            ボールの位置座標 (x, y)
        left : TeamPos
            左チームの各選手の位置座標のリスト
        right : TeamPos
            右チームの各選手の位置座標のリスト

        """
        self.ball = ball
        self.left = left
        self.right = right

    @classmethod
    def read_pos(cls, pos: List[Union[Pos, TeamPos]]) -> "PosData":
        return PosData(
            typing.cast(Pos, pos[0]),
            typing.cast(TeamPos, pos[1]),
            typing.cast(TeamPos, pos[2]),
        )

    @staticmethod
    def reverse(pos: Pos) -> Pos:
        """
        位置座標を反転します。
        
        Parameters
        ----------
        pos : Pos
            反転する位置座標 (x, y)
        
        Returns
        -------
        Pos
            反転された位置座標 (-x, -y)
        """
        x, y = pos
        return (-x, -y)
    
    @property
    def pos(self) -> List[Union[Pos, TeamPos]]:
        return [self.ball, self.left, self.right]

    def reverse_sides(self) -> None:
        """
        左右のチームの位置座標を反転します。
        """
        self.ball = PosData.reverse(self.ball)
        self.left, self.right = self.right, self.left
        self.left = [PosData.reverse(pos) for pos in self.left]
        self.right = [PosData.reverse(pos) for pos in self.right]

    @property
    def flat_pos(self) -> List[float]:
        """
        すべての位置座標をフラットなリストとして取得します。
        
        Returns
        -------
        List[float]
            ボールと両チームの選手の位置座標をフラットにしたリスト
        """
        ball_flat: List[float] = list(self.ball)
        left_flat: List[float] = list(itertools.chain.from_iterable(self.left))
        right_flat: List[float] = list(itertools.chain.from_iterable(self.right))
        return ball_flat + left_flat + right_flat