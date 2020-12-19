use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::ops::{Index, IndexMut};
use std::{cell::RefCell, thread::current};
use std::{
    cmp::Ordering,
    collections::{HashSet, VecDeque},
};
use std::{collections::HashMap, io};

const VIEW_RANGE: i32 = 2;

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)]
struct TilePos(i32, i32);

#[derive(PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn values() -> [Direction; 4] {
        [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ]
    }
}

impl TilePos {
    fn distance(&self, other: &TilePos) -> i32 {
        let dx = self.0 - other.0;
        let dy = self.1 - other.1;
        dx.abs() + dy.abs()
    }

    fn neighbors(&self) -> [TilePos; 4] {
        [
            TilePos(self.0 + 1, self.1),
            TilePos(self.0 - 1, self.1),
            TilePos(self.0, self.1 + 1),
            TilePos(self.0, self.1 - 1),
        ]
    }

    fn mov(&self, dir: Direction) -> TilePos {
        match dir {
            Direction::Left => TilePos(self.0 - 1, self.1),
            Direction::Right => TilePos(self.0 + 1, self.1),
            Direction::Up => TilePos(self.0, self.1 - 1),
            Direction::Down => TilePos(self.0, self.1 + 1),
        }
    }
}

impl Distribution<Direction> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Direction {
        match rng.gen_range(0, 4) {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            _ => Direction::Right,
        }
    }
}

#[derive(PartialEq, Copy, Clone)]
enum Tile {
    Start,
    ControlRoom,
    Hollow,
    Wall,
    Unknown,
}

impl Tile {
    fn parse(c: char) -> Tile {
        match c {
            'T' => Tile::Start,
            'C' => Tile::ControlRoom,
            '.' => Tile::Hollow,
            '#' => Tile::Wall,
            '?' => Tile::Unknown,
            _ => panic!(),
        }
    }

    fn unparse(&self) -> char {
        match self {
            Tile::Start => 'T',
            Tile::ControlRoom => 'C',
            Tile::Hollow => '.',
            Tile::Wall => '#',
            Tile::Unknown => '?',
            _ => panic!(),
        }
    }
}

fn tile_direction(from: &TilePos, to: &TilePos) -> Direction {
    let dx = (from.0 - to.0).abs();
    let dy = (from.1 - to.1).abs();
    let sx = (from.0 - to.0).signum();
    let sy = (from.1 - to.1).signum();
    if dx < dy {
        if sy > 0 {
            Direction::Up
        } else {
            Direction::Down
        }
    } else {
        if sx > 0 {
            Direction::Left
        } else {
            Direction::Right
        }
    }
}

struct Map {
    tiles: Vec<Vec<Tile>>,
}

impl Map {
    fn new(width: i32, height: i32) -> Map {
        Map {
            tiles: vec![vec![Tile::Unknown; width as usize]; height as usize],
        }
    }

    fn parse<T: AsRef<str>>(input: &[T]) -> Map {
        Map {
            tiles: input
                .iter()
                .map(|line| line.as_ref().chars().map(Tile::parse).collect())
                .collect(),
        }
    }

    fn row_str(&self, row: i32) -> String {
        let mut result = String::new();
        for i in 0..self.width() {
            result.push(self[TilePos(i, row)].unparse());
        }
        result
    }

    fn width(&self) -> i32 {
        self.tiles[0].len() as i32
    }

    fn height(&self) -> i32 {
        self.tiles.len() as i32
    }

    fn iter(&self) -> MapIterator {
        MapIterator {
            map: &self,
            x: 0,
            y: 0,
        }
    }

    fn is_inside(&self, pos: TilePos) -> bool {
        !(pos.0 < 0 || pos.0 >= self.width() || pos.1 < 0 || pos.1 >= self.height())
    }

    fn look_for_tile(&self, tile: Tile) -> Option<TilePos> {
        for x in 0..self.width() {
            for y in 0..self.height() {
                if self[TilePos(x, y)] == tile {
                    return Some(TilePos(x, y));
                }
            }
        }
        None
    }
}

struct MapIterator<'a> {
    map: &'a Map,
    x: i32,
    y: i32,
}

impl<'a> Iterator for MapIterator<'a> {
    type Item = (TilePos, Tile);

    fn next(&mut self) -> Option<(TilePos, Tile)> {
        if self.y >= self.map.height() {
            return None;
        }
        let pos = TilePos(self.x, self.y);
        self.x += 1;
        if self.x >= self.map.width() {
            self.x = 0;
            self.y += 1;
        }
        Some((pos, self.map[pos]))
    }
}

impl Index<TilePos> for Map {
    type Output = Tile;

    fn index(&self, index: TilePos) -> &Tile {
        &self.tiles[index.1 as usize][index.0 as usize]
    }
}

impl IndexMut<TilePos> for Map {
    fn index_mut(&mut self, index: TilePos) -> &mut Tile {
        &mut self.tiles[index.1 as usize][index.0 as usize]
    }
}

fn bfs_reachable(from: TilePos, map: &Map) -> HashSet<TilePos> {
    let mut visited = HashSet::new();
    let mut visit_queue = VecDeque::new();
    visit_queue.push_back(from);
    while !visit_queue.is_empty() {
        let current = visit_queue.pop_front().unwrap();
        if visited.contains(&current) {
            continue;
        }
        visited.insert(current);
        for next in current.neighbors().iter() {
            if !map.is_inside(*next) {
                continue;
            }
            if map[*next] == Tile::Start
                || map[*next] == Tile::ControlRoom
                || map[*next] == Tile::Hollow
            {
                visit_queue.push_back(*next);
            }
        }
    }
    visited
}

fn bfs(from: TilePos, to: TilePos, map: &Map) -> Option<Vec<TilePos>> {
    if from == to {
        return Some(vec![from]);
    }
    if map[from] == Tile::Wall || map[to] == Tile::Wall {
        return None;
    }
    let mut visited = HashSet::new();
    let mut predecessors = HashMap::<TilePos, TilePos>::new();
    let mut visit_queue = VecDeque::new();
    visit_queue.push_back(from);
    loop {
        let current = visit_queue.pop_front()?;
        if visited.contains(&current) {
            continue;
        }
        visited.insert(current);
        if current == to {
            let mut path = vec![current];
            loop {
                let backwards = *path.last().unwrap();
                if from == backwards {
                    return Some(path.into_iter().rev().collect());
                }
                let predec = predecessors[&backwards];
                path.push(predec);
            }
        }
        for next in current.neighbors().iter() {
            if !map.is_inside(*next) {
                continue;
            }
            if visited.contains(next) {
                continue;
            }
            if map[*next] == Tile::Start
                || map[*next] == Tile::ControlRoom
                || map[*next] == Tile::Hollow
            {
                visit_queue.push_back(*next);
                predecessors.insert(*next, current);
            }
        }
    }
}

fn bfs_directions(from: TilePos, to: TilePos, map: &Map) -> Option<Vec<Direction>> {
    let positions = bfs(from, to, map)?;
    let x = positions
        .iter()
        .skip(1)
        .zip(positions.iter())
        .map(|(a, b)| tile_direction(b, a));
    Some(x.collect())
}

struct GameMemory {
    way_back: bool,
    visited: HashSet<TilePos>,
    current_target: TilePos,
    last_number_of_hidden_tiles: i32,
}

impl GameMemory {
    fn new() -> GameMemory {
        GameMemory {
            way_back: false,
            visited: HashSet::new(),
            current_target: TilePos(0, 0),
            last_number_of_hidden_tiles: -1,
        }
    }

    fn update(&mut self, state: &GameState) {
        if state.map[state.player_pos] == Tile::ControlRoom {
            self.way_back = true;
        }
        self.visited.insert(state.player_pos);
        let unknowns = state
            .map
            .iter()
            .filter(|(_, tile)| *tile == Tile::Unknown)
            .count() as i32;
        if unknowns != self.last_number_of_hidden_tiles || state.player_pos == self.current_target {
            self.last_number_of_hidden_tiles = unknowns;
            self.current_target = self.find_target(state)
        }
    }

    fn find_target(&self, state: &GameState) -> TilePos {
        let reachable_tiles = bfs_reachable(state.player_pos, &state.map)
            .into_iter()
            .collect::<Vec<_>>();
        if state.map.look_for_tile(Tile::ControlRoom).is_some() {
            print!("");
        }
        let tile_ordering = |lhs: &TilePos, rhs: &TilePos| -> Ordering {
            // Equality of tiles
            if lhs == rhs {
                return Ordering::Equal;
            }

            // If the end goal is reachable, do so.
            let goal_tile = if self.way_back {
                Tile::Start
            } else {
                Tile::ControlRoom
            };
            if state.map[*lhs] == goal_tile {
                return Ordering::Less;
            } else if state.map[*rhs] == goal_tile {
                return Ordering::Greater;
            }

            // Prefer non-visited over visited tiles.
            if self.visited.contains(lhs) && !self.visited.contains(rhs) {
                return Ordering::Greater;
            } else if !self.visited.contains(lhs) && self.visited.contains(rhs) {
                return Ordering::Less;
            }

            // Avoid the current player position.
            if *lhs == state.player_pos {
                return Ordering::Greater;
            } else if *rhs == state.player_pos {
                return Ordering::Less;
            }

            // Choose the tile with more unknown tiles nearby.
            let lhs_unknowns = state
                .map
                .iter()
                .filter(|(pos, tile)| lhs.distance(pos) <= VIEW_RANGE && *tile == Tile::Unknown)
                .count();
            let rhs_unknowns = state
                .map
                .iter()
                .filter(|(pos, tile)| rhs.distance(pos) <= VIEW_RANGE && *tile == Tile::Unknown)
                .count();
            if lhs_unknowns < rhs_unknowns {
                return Ordering::Greater;
            } else if lhs_unknowns > rhs_unknowns {
                return Ordering::Less;
            }

            // Choose the nearer tile.
            let dist_cmp = state
                .player_pos
                .distance(lhs)
                .partial_cmp(&state.player_pos.distance(rhs))
                .unwrap();
            if dist_cmp != Ordering::Equal {
                return dist_cmp;
            }

            // For determinism.
            lhs.partial_cmp(rhs).unwrap()
        };
        reachable_tiles.into_iter().min_by(tile_ordering).unwrap()
    }
}

struct GameState {
    player_pos: TilePos,
    map: Map,
}

impl GameState {
    fn next_move(&self, memory: &GameMemory) -> Direction {
        let bfs_moves: HashMap<_, _> = Direction::values()
            .into_iter()
            .map(|&dir| (dir, self.player_pos.mov(dir)))
            .filter(|(_, pos)| self.map.is_inside(*pos))
            .map(|(dir, pos)| (dir, bfs_directions(pos, memory.current_target, &self.map)))
            .collect();
        let rate_direction = |lhs: &Direction, rhs: &Direction| -> Ordering {
            let lhs_bfs = &bfs_moves[lhs];
            let rhs_bfs = &bfs_moves[rhs];
            if lhs_bfs.is_none() {
                if rhs_bfs.is_none() {
                    return Ordering::Equal;
                } else {
                    return Ordering::Greater;
                }
            } else if rhs_bfs.is_none() {
                return Ordering::Less;
            }

            let lhs_bfs = lhs_bfs.as_ref().unwrap();
            let rhs_bfs = rhs_bfs.as_ref().unwrap();
            if lhs_bfs.len() < rhs_bfs.len() {
                return Ordering::Less;
            } else if lhs_bfs.len() > rhs_bfs.len() {
                return Ordering::Greater;
            }

            let lhs_next = self.player_pos.mov(*lhs);
            let rhs_next = self.player_pos.mov(*rhs);
            let lhs_unknowns = self
                .map
                .iter()
                .filter(|(pos, tile)| {
                    lhs_next.distance(pos) <= VIEW_RANGE && *tile == Tile::Unknown
                })
                .count();
            let rhs_unknowns = self
                .map
                .iter()
                .filter(|(pos, tile)| {
                    rhs_next.distance(pos) <= VIEW_RANGE && *tile == Tile::Unknown
                })
                .count();
            if lhs_unknowns < rhs_unknowns {
                return Ordering::Greater;
            } else if lhs_unknowns > rhs_unknowns {
                return Ordering::Less;
            }

            // For determinism.
            lhs.partial_cmp(rhs).unwrap()
        };
        *bfs_moves
            .keys()
            .min_by(|a, b| rate_direction(a, b))
            .unwrap()
    }
}

// Return: number of rows, number of columns, number of rounds between the time the alarm countdown is activated and the time the alarm goes off.
fn parse_initial_input(input_line: &str) -> (i32, i32, i32) {
    let input = input_line.split(" ").collect::<Vec<_>>();
    let r = parse_input!(input[0], i32);
    let c = parse_input!(input[1], i32);
    let a = parse_input!(input[2], i32);
    (r, c, a)
}

// Return: Player coordinates, map
fn parse_loop_input<T: AsRef<str>>(input_lines: &[T]) -> (TilePos, Map) {
    let first_line = input_lines[0].as_ref().split(" ").collect::<Vec<_>>();
    let kr = parse_input!(first_line[0], i32);
    let kc = parse_input!(first_line[1], i32);
    let map = Map::parse(&input_lines[1..]);
    (TilePos(kc, kr), map)
}

fn one_game_step<T: AsRef<str>>(
    r: i32,
    c: i32,
    a: i32,
    memory: &mut GameMemory,
    input: &[T],
    output_func: &mut dyn FnMut(&str),
) {
    let (player_pos, map) = parse_loop_input(&input);
    let game_state = GameState { player_pos, map };
    memory.update(&game_state);
    match game_state.next_move(memory) {
        Direction::Up => output_func("UP"),
        Direction::Down => output_func("DOWN"),
        Direction::Left => output_func("LEFT"),
        Direction::Right => output_func("RIGHT"),
    };
}

// Read n lines from input
fn read_n_lines_from_stdin(n: usize) -> Vec<String> {
    (0..n as usize)
        .map(|_| {
            let mut input_line = String::new();
            io::stdin().read_line(&mut input_line).unwrap();
            input_line.trim().to_string()
        })
        .collect::<Vec<String>>()
}

fn generic_main(
    game_over: &mut dyn FnMut() -> bool,
    read_n_lines: &mut dyn FnMut(usize) -> Vec<String>,
    output: &mut dyn FnMut(&str),
) {
    let (r, c, a) = parse_initial_input(&read_n_lines(1)[0]);
    let mut memory = GameMemory::new();
    while !game_over() {
        one_game_step(r, c, a, &mut memory, &read_n_lines(1 + r as usize), output);
    }
}

/*fn main() {
    generic_main(&mut || false, &mut read_n_lines_from_stdin, &mut |x| println!("{}", x));
}*/

////////// Simulation
///

fn uncover_map(full_map: &Map, current_map: &mut Map, base_pos: TilePos, range: i32) {
    for x in -range..=range {
        for y in -range..=range {
            let pos = TilePos(base_pos.0 + x, base_pos.1 + y);
            if current_map.is_inside(pos) {
                current_map[pos] = full_map[pos];
            }
        }
    }
}

fn print_map_with_highlight(map: &Map, highlight: TilePos) {
    for y in 0..map.height() {
        for x in 0..map.width() {
            if highlight.0 == x && highlight.1 == y {
                print!(
                    "{}{}{}",
                    "\x1b[0;31m",
                    map[TilePos(x, y)].unparse(),
                    "\x1b[0m"
                );
            } else {
                print!("{}", map[TilePos(x, y)].unparse());
            }
        }
        println!();
    }
}

/// Returns None() if the puzzle wasn't solved or Some(fuel_left).
fn simulate_main(alarm: i32, map_str: &[&str]) -> Option<i32> {
    let full_map = Map::parse(map_str);
    let current_map_cell = RefCell::new(Map::new(full_map.width(), full_map.height()));
    let player_pos_cell = RefCell::new(full_map.look_for_tile(Tile::Start).unwrap());
    let control_reached_cell = RefCell::new(false);
    let fuel_cell = RefCell::new(1200);
    let alarm_cell = RefCell::new(alarm);

    let mut read_n_lines_counter = 0usize;
    let mut read_n_lines = |n| {
        let mut lines = Vec::new();
        for i in 0..n {
            let j = (i + read_n_lines_counter) as i32;
            let line = if j == 0 {
                format!(
                    "{} {} {}",
                    full_map.height(),
                    full_map.width(),
                    *alarm_cell.borrow()
                )
            } else {
                let current_map = current_map_cell.borrow();
                let player_pos = player_pos_cell.borrow();
                let k = (j - 1) % (current_map.height() + 1);
                if k == 0 {
                    format!("{} {}", player_pos.1, player_pos.0)
                } else {
                    current_map.row_str(k - 1)
                }
            };
            lines.push(line);
        }
        read_n_lines_counter += n;
        if read_n_lines_counter == 0 {}
        lines
    };

    let mut output = |answer: &str| {
        let mut player_pos = player_pos_cell.borrow_mut();
        let mut current_map = current_map_cell.borrow_mut();
        match answer {
            "UP" => player_pos.1 -= 1,
            "DOWN" => player_pos.1 += 1,
            "LEFT" => player_pos.0 -= 1,
            "RIGHT" => player_pos.0 += 1,
            _ => panic!(),
        }
        uncover_map(&full_map, &mut current_map, *player_pos, VIEW_RANGE);
        let control_reached = &mut *control_reached_cell.borrow_mut();
        *control_reached = *control_reached || current_map[*player_pos] == Tile::ControlRoom;
        *fuel_cell.borrow_mut() -= 1;
        if *control_reached {
            *alarm_cell.borrow_mut() -= 1;
        }
        print!("\x1B[2J\x1B[1;1H");
        print_map_with_highlight(&*current_map, *player_pos);
        println!();
    };

    let mut game_over = || {
        let player_pos = player_pos_cell.borrow();
        let current_map = current_map_cell.borrow();
        let control_reached = control_reached_cell.borrow();
        let goal_reached = *control_reached && current_map[*player_pos] == Tile::Start;
        let fuel_empty = *fuel_cell.borrow() <= 0;
        let alarm_empty = *alarm_cell.borrow() <= 0;
        let wall_hit = current_map[*player_pos] == Tile::Wall;
        goal_reached || fuel_empty || alarm_empty || wall_hit
    };

    uncover_map(
        &full_map,
        &mut current_map_cell.borrow_mut(),
        *player_pos_cell.borrow(),
        VIEW_RANGE,
    );
    print!("\x1B[2J\x1B[1;1H");
    print_map_with_highlight(&*current_map_cell.borrow(), *player_pos_cell.borrow());
    println!();
    generic_main(&mut game_over, &mut read_n_lines, &mut output);

    if *player_pos_cell.borrow() == full_map.look_for_tile(Tile::Start).unwrap()
        && *control_reached_cell.borrow()
    {
        Some(*fuel_cell.borrow())
    } else {
        None
    }
}

const TASK1: ([&str; 3], i32) = (["##########", "#T......C#", "##########"], 8);
const TASK1B: ([&str; 1], i32) = (["T......C"], 8);
const TASK2: ([&str; 15], i32) = (
    [
        "#####################.....####",
        "########........#.############",
        "#################.#......T####",
        "############...#..#.#######.##",
        "######...##.#######...#####.##",
        "######...##.#C..#####.#####.##",
        "######...##.###.......########",
        "###########.#.########.......#",
        "#######..##...##....##########",
        "#######################..#####",
        "######.....############..#####",
        "######.....############.######",
        "######.....#########....######",
        "#####################..#######",
        "##############################",
    ],
    22,
);
const TASK3: ([&str; 15], i32) = (
    [
        "##############################",
        "#T...........................#",
        "##...........................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#............................#",
        "#...........................C#",
        "##############################",
    ],
    40,
);
const TASK4: ([&str; 15], i32) = (
    [
        "##############################",
        "##.###########.###############",
        "#...#.######.#......#......T##",
        "###.#.######.#.######.########",
        "##...........#.######.########",
        "###.#.######......###.########",
        "#...#.###....#.##.###.########",
        "#.############.##.....########",
        "#......##......###############",
        "###.###############..........#",
        "###.#####......####.########.#",
        "###.......####............####",
        "##############.######.###.##.#",
        "####C..........###....###....#",
        "##############################",
    ],
    69,
);
const TASK5: ([&str; 15], i32) = (
    [
        "##############################",
        "#.##.....#..........T........#",
        "#.########.#################.#",
        "#.##.......................#.#",
        "#.##.##########.##########.#.#",
        "#.##.......................#.#",
        "#.########.#########.#######.#",
        "#.########...........#######.#",
        "#.########.#########.#######.#",
        "#.##.......................#.#",
        "#.##.##########.##########.#.#",
        "#.##.......................#.#",
        "#.########.#########.#######.#",
        "#.##.....#C###########.....#.#",
        "##############################",
    ],
    43,
);
const TASK6: ([&str; 15], i32) = (
    [
        "##############################",
        "##.##....#.##.....##.###.....#",
        "##....##......####.......###.#",
        "#..##.##.#.##.####.##.##.###.#",
        "##.##.....#........##C.......#",
        "#.....###...###......###.#####",
        "###.#.###.#.....##..........##",
        "#.........###.###.##........##",
        "###.###.#.............##.##..#",
        "###.....###.###.#........##.##",
        "##.#######...##.##.##.#.#...##",
        "#.............###..#.#......##",
        "#.####.######.###.#######...##",
        "###........T#.......##......##",
        "##############################",
    ],
    34,
);

fn run_all() {
    assert!(simulate_main(TASK1.1, &TASK1.0).is_some());
    assert!(simulate_main(TASK1B.1, &TASK1B.0).is_some());
    assert!(simulate_main(TASK2.1, &TASK2.0).is_some());
    assert!(simulate_main(TASK3.1, &TASK3.0).is_some());
    assert!(simulate_main(TASK4.1, &TASK4.0).is_some());
    assert!(simulate_main(TASK5.1, &TASK5.0).is_some());
    assert!(simulate_main(TASK6.1, &TASK6.0).is_some());
}

fn main() {
    let task = TASK6;
    let result = simulate_main(task.1, &task.0);
    match result {
        None => println!("Could not solve."),
        Some(i) => println!("Solved with {} fuel left.", i),
    }
//    run_all();
}
