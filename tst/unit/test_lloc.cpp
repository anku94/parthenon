//
// Created by Ankush J on 5/15/23.
//

#include <catch2/catch.hpp>

#include "defs.hpp"

TEST_CASE("LogicalLocation order", "[LogicalLocation]") {
    SECTION("Parent computation works") {
        parthenon::LogicalLocation a{8, 8, 8, 3},
        a_par_actual{4, 4, 4, 2},
        a_par;
        int offset;

        parthenon::LogicalLocation::GetParent(a, a_par, offset);
        REQUIRE(a_par == a_par_actual);
        REQUIRE(offset == 0);


        a = {9, 9, 9, 3};
        parthenon::LogicalLocation::GetParent(a, a_par, offset);
        REQUIRE(a_par == a_par_actual);
        REQUIRE(offset == 7);
    }

    SECTION("SortComparatorWorks") {
        parthenon::LogicalLocation a{8, 8, 8, 4},
        b{8, 9, 9, 4}, c{3, 3, 3, 3}, d{10, 10, 10, 4};

        REQUIRE(parthenon::LogicalLocation::SortComparator(a, b));
        REQUIRE(!parthenon::LogicalLocation::SortComparator(b, a));
        REQUIRE(parthenon::LogicalLocation::SortComparator(a, d));
        REQUIRE(parthenon::LogicalLocation::SortComparator(b, d));
        REQUIRE(parthenon::LogicalLocation::SortComparator(c, a));

        std::vector<parthenon::LogicalLocation> llv = {a, b, c, d};
        std::sort(llv.begin(), llv.end(), parthenon::LogicalLocation::SortComparator);

        REQUIRE(llv == std::vector<parthenon::LogicalLocation>{c, a, b, d});
    }
}