package actions;

public class BitTigerTest {
    BitTiger test = new BitTiger();

    @org.junit.Test
    public void testMethod() {
        BitTiger.LRUCache lru = new BitTiger.LRUCache(1);
        lru.put(2, 1);
        lru.get(2);
        lru.put(3, 2);
        int result = test.largestRectangleArea(new int[]{2, 1, 2});
        test.minWindow("ADOBECODEBANC", "ABC");
        test.search(new int[]{3, 1}, 3);
    }
}
